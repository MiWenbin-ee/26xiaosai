from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import utime
import image
import random
import gc
import sys
import aidemo
from machine import UART

# K230 UART parity uses integer values. 0 means no parity.
UART_PARITY_NONE = 0

# State constants
STATE_IDLE = 0
STATE_CALIBRATING = 1
STATE_WAIT_MEAS = 2
STATE_MEASURING = 3


def init_uart2():
    try:
        return UART(2, baudrate=115200, bits=8, parity=UART_PARITY_NONE, stop=1)
    except TypeError:
        return UART(2, 115200)


class SegmentationApp(AIBase):
    def __init__(self, kmodel_path, labels, model_input_size,
                 confidence_threshold=0.2, nms_threshold=0.5, mask_threshold=0.5,
                 rgb888p_size=[224, 224], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.mask_threshold = mask_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.color_four = [(255, 220, 20, 60), (255, 119, 11, 32), (255, 0, 0, 142), (255, 0, 0, 230),
                           (255, 106, 0, 228), (255, 0, 60, 100), (255, 0, 80, 100), (255, 0, 0, 70),
                           (255, 0, 0, 192), (255, 250, 170, 30), (255, 100, 170, 30), (255, 220, 220, 0),
                           (255, 175, 116, 175), (255, 250, 0, 30), (255, 165, 42, 42), (255, 255, 77, 255),
                           (255, 0, 226, 252), (255, 182, 182, 255), (255, 0, 82, 0), (255, 120, 166, 157)]
        self.masks = np.zeros((1, self.display_size[1], self.display_size[0], 4))
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

        # Distance polynomial coefficients
        self.dist_a = -0.00000873
        self.dist_b = 0.01196712
        self.dist_c = -5.65386753
        self.dist_d = 964.76871412

        # Height calibration coefficient
        self.k_H = 0.00091

        # State machine
        self.state = STATE_IDLE

        # Measurement data collection
        self.meas_buffer_D = []
        self.meas_buffer_H = []
        self.meas_samples_target = 10
        self.meas_total_frames = 0
        self.meas_done = False
        self.last_D = 0.0
        self.last_H = 0.0

        # Sliding window for smoothing display
        self.window_size = 10
        self.history_bbox = []
        self.history_D = []

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            seg_res = aidemo.segment_postprocess(results,
                                                  [self.rgb888p_size[1], self.rgb888p_size[0]],
                                                  self.model_input_size,
                                                  [self.display_size[1], self.display_size[0]],
                                                  self.confidence_threshold, self.nms_threshold,
                                                  self.mask_threshold, self.masks)
            return seg_res

    def compute_D(self, bottom_y):
        a, b, c, d = self.dist_a, self.dist_b, self.dist_c, self.dist_d
        return a * (bottom_y ** 3) + b * (bottom_y ** 2) + c * bottom_y + d

    def compute_H(self, h_pixel, D_real):
        return h_pixel * D_real * self.k_H

    def check_uart(self, uart):
        if uart is None:
            return
        if uart.any():
            try:
                line = uart.readline()
                if line:
                    cmd = line.decode().strip()
                    if cmd == "CAL_START":
                        self.state = STATE_CALIBRATING
                        self.history_bbox.clear()
                        self.history_D.clear()
                    elif cmd == "CAL_END":
                        self.state = STATE_WAIT_MEAS
                    elif cmd.startswith("K:"):
                        self.k_H = float(cmd[2:])
                    elif cmd == "MEAS_START":
                        self.state = STATE_MEASURING
                        self.meas_buffer_D = []
                        self.meas_buffer_H = []
                        self.meas_total_frames = 0
                        self.meas_done = False
                        self.history_bbox.clear()
                        self.history_D.clear()
                    elif cmd == "RET_IDLE":
                        self.state = STATE_WAIT_MEAS
            except:
                pass

    def is_bottle_in_center(self, x, y, w, h):
        cx = x + w / 2
        cy = y + h / 2
        dw = self.display_size[0]
        dh = self.display_size[1]
        left = dw / 3
        right = 2 * dw / 3
        top = dh / 3
        bottom = 2 * dh / 3
        return left <= cx <= right and top <= cy <= bottom

    def filter_outliers_mad(self, data):
        n = len(data)
        if n < 3:
            return data

        sorted_data = sorted(data)
        median_val = sorted_data[n // 2]

        abs_dev = [abs(x - median_val) for x in data]
        abs_dev_sorted = sorted(abs_dev)
        mad = abs_dev_sorted[len(abs_dev) // 2]

        if mad < 0.0001:
            return data

        threshold = 3.0 * mad
        filtered = [x for x in data if abs(x - median_val) <= threshold]

        if len(filtered) < 2:
            return data
        return filtered

    def _draw_state_indicator(self, pl):
        state_names = {STATE_IDLE: "IDLE", STATE_CALIBRATING: "CAL",
                       STATE_WAIT_MEAS: "WAIT", STATE_MEASURING: "MEAS"}
        name = state_names.get(self.state, "???")
        color_map = {STATE_IDLE: (255, 128, 128, 128),
                     STATE_CALIBRATING: (255, 255, 255, 0),
                     STATE_WAIT_MEAS: (255, 0, 255, 0),
                     STATE_MEASURING: (255, 0, 128, 255)}
        color = color_map.get(self.state, (255, 128, 128, 128))
        pl.osd_img.draw_string_advanced(5, 5, 28, "S:" + name, color=color)

        if self.state == STATE_CALIBRATING:
            k_text = "K:%.6f" % self.k_H
            pl.osd_img.draw_string_advanced(5, 35, 24, k_text, color=(255, 255, 255, 0))

    def draw_result(self, pl, seg_res, uart):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            self.check_uart(uart)

            pl.osd_img.clear()

            if seg_res[0]:
                mask_img = image.Image(self.display_size[0], self.display_size[1],
                                       image.ARGB8888, alloc=image.ALLOC_REF, data=self.masks)
                pl.osd_img.copy_from(mask_img)

                dets, ids, scores = seg_res[0], seg_res[1], seg_res[2]
                bottle_detected = False

                for i, det in enumerate(dets):
                    class_id = int(ids[i])
                    class_name = self.labels[class_id]
                    raw_x1, raw_y1, raw_w, raw_h = det[0], det[1], det[2], det[3]

                    if class_name == "bottle":
                        bottle_detected = True

                        # Sliding window smoothing
                        self.history_bbox.append([raw_x1, raw_y1, raw_w, raw_h])
                        if len(self.history_bbox) > self.window_size:
                            self.history_bbox.pop(0)

                        num = len(self.history_bbox)
                        s_x1 = int(sum(b[0] for b in self.history_bbox) / num)
                        s_y1 = int(sum(b[1] for b in self.history_bbox) / num)
                        s_w = int(sum(b[2] for b in self.history_bbox) / num)
                        s_h = int(sum(b[3] for b in self.history_bbox) / num)

                        bottom_x = s_x1 + s_w // 2
                        bottom_y = s_y1 + s_h

                        current_D = self.compute_D(bottom_y)

                        self.history_D.append(current_D)
                        if len(self.history_D) > self.window_size:
                            self.history_D.pop(0)

                        final_D = sum(self.history_D) / len(self.history_D)
                        final_H = self.compute_H(s_h, final_D)

                        self.last_D = final_D
                        self.last_H = final_H

                        if self.state == STATE_MEASURING:
                            self.meas_total_frames += 1
                            center_ok = self.is_bottle_in_center(s_x1, s_y1, s_w, s_h)

                            if center_ok:
                                pl.osd_img.draw_rectangle(s_x1, s_y1, s_w, s_h,
                                                          color=(255, 0, 255, 0), thickness=3)
                                pl.osd_img.draw_circle(bottom_x, bottom_y, 8,
                                                       color=(255, 255, 0, 0), thickness=2, fill=True)

                                self.meas_buffer_D.append(final_D)
                                self.meas_buffer_H.append(final_H)

                                progress = len(self.meas_buffer_D)
                                info = "MEAS:%d/%d" % (progress, self.meas_samples_target)
                                pl.osd_img.draw_string_advanced(s_x1, s_y1 - 40, 35, info,
                                                                color=(255, 0, 255, 0))

                                if progress >= self.meas_samples_target:
                                    fD = self.filter_outliers_mad(self.meas_buffer_D)
                                    fH = self.filter_outliers_mad(self.meas_buffer_H)

                                    self.last_D = sum(fD) / len(fD)
                                    self.last_H = sum(fH) / len(fH)
                                    self.meas_done = True
                            else:
                                pl.osd_img.draw_rectangle(s_x1, s_y1, s_w, s_h,
                                                          color=(255, 255, 0, 0), thickness=2)
                                hint = "Move to center"
                                pl.osd_img.draw_string_advanced(s_x1, s_y1 - 40, 30, hint,
                                                                color=(255, 255, 255, 0))

                            if self.meas_total_frames > 120:
                                self.state = STATE_WAIT_MEAS
                                self.meas_buffer_D = []
                                self.meas_buffer_H = []
                                self.meas_done = False
                        else:
                            # IDLE / CALIBRATING / WAIT_MEAS: normal live display
                            pl.osd_img.draw_rectangle(s_x1, s_y1, s_w, s_h,
                                                      color=(255, 0, 255, 0), thickness=2)
                            pl.osd_img.draw_circle(bottom_x, bottom_y, 8,
                                                   color=(255, 255, 0, 0), thickness=2, fill=True)

                            text = "D:%.1fcm H:%.1fcm" % (final_D, final_H)
                            pl.osd_img.draw_string_advanced(s_x1, s_y1 - 40, 35, text,
                                                            color=(255, 0, 0, 255))
                    else:
                        x1, y1, w, h = map(int, det)
                        pl.osd_img.draw_string_advanced(x1, y1 - 40, 32,
                                                        " " + class_name + " " + str(round(scores[i], 2)),
                                                        color=self.get_color(class_id))

                if not bottle_detected:
                    self.history_bbox.clear()
                    self.history_D.clear()
            else:
                self.history_bbox.clear()
                self.history_D.clear()
                if self.state == STATE_MEASURING:
                    self.meas_total_frames += 1

            self._draw_state_indicator(pl)

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = float(dst_w) / self.rgb888p_size[0]
        ratio_h = float(dst_h) / self.rgb888p_size[1]
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(dh - 0.1))
        bottom = (int)(round(dh + 0.1))
        left = (int)(round(dw - 0.1))
        right = (int)(round(dw + 0.1))
        return top, bottom, left, right

    def get_color(self, x):
        idx = x % len(self.color_four)
        return self.color_four[idx]


if __name__ == "__main__":
    display_mode = "lcd"
    if display_mode == "hdmi":
        display_size = [1920, 1080]
    else:
        display_size = [800, 480]

    kmodel_path = "/sdcard/app/yolov8n_320.kmodel"
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
              "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
              "toothbrush"]

    confidence_threshold = 0.2
    nms_threshold = 0.5
    mask_threshold = 0.5
    rgb888p_size = [320, 320]

    uart = init_uart2()

    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create(hmirror=True, vflip=True)
    seg = SegmentationApp(kmodel_path, labels=labels, model_input_size=[320, 320],
                          confidence_threshold=confidence_threshold, nms_threshold=nms_threshold,
                          mask_threshold=mask_threshold, rgb888p_size=rgb888p_size,
                          display_size=display_size, debug_mode=0)
    seg.config_preprocess()

    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total", 1):
                img = pl.get_frame()
                seg_res = seg.run(img)
                seg.draw_result(pl, seg_res, uart)
                pl.show_image()

                if seg.meas_done:
                    result_str = "D:%.1f,H:%.1f\n" % (seg.last_D, seg.last_H)
                    uart.write(result_str)
                    seg.meas_done = False
                    seg.state = STATE_WAIT_MEAS

                gc.collect()
    except Exception as e:
        sys.print_exception(e)
    finally:
        seg.deinit()
        pl.destroy()
