import time
# 【量产版核心 1】开机雷打不动先睡 3 秒！
# 让电容充满，让摄像头传感器彻底稳住，坚决不抢跑
time.sleep(3)

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import utime
import image
import gc
import sys
import aidemo
from machine import UART, FPIOA

UART_PARITY_NONE = 0

STATE_IDLE = 0
STATE_CALIBRATING = 1
STATE_WAIT_MEAS = 2
STATE_MEASURING = 3

CAL_FILE = "/sdcard/app/cal_data.json"
CAL_TARGETS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

def init_uart2():
    try:
        fpioa = FPIOA()
        fpioa.set_function(5, FPIOA.UART2_TXD)
        fpioa.set_function(6, FPIOA.UART2_RXD)
        return UART(2, baudrate=115200, bits=8, parity=UART_PARITY_NONE, stop=1)
    except:
        return None

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
        self.color_four = [(255, 20, 60), (255, 11, 32), (255, 0, 142)]

        self.masks = np.zeros((1, self.display_size[1], self.display_size[0], 4), dtype=np.uint8)
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

        self.dist_a, self.dist_b, self.dist_c, self.dist_d = -0.00000873, 0.01196712, -5.65386753, 964.76871412
        self.k_H = 0.00091

        self.state = STATE_IDLE
        self.meas_buffer_D = []
        self.meas_buffer_H = []
        self.meas_samples_target = 10
        self.meas_total_frames = 0
        self.meas_done = False
        self.last_D, self.last_H = 0.0, 0.0
        self.window_size = 10
        self.history_bbox, self.history_D = [], []
        self.cal_targets = CAL_TARGETS
        self.cal_index = 0
        self.cal_lut = []
        self.cal_done = False

        self.ui_popup_frames = 0
        self.ui_popup_text = ""
        self.ui_popup_color = (255, 255, 255, 0)
        self.miss_count = 0

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", False):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", False):
            return aidemo.segment_postprocess(results, [self.rgb888p_size[1], self.rgb888p_size[0]],
                                              self.model_input_size, [self.display_size[1], self.display_size[0]],
                                              self.confidence_threshold, self.nms_threshold,
                                              self.mask_threshold, self.masks)

    def compute_D(self, bottom_y):
        a, b, c, d = self.dist_a, self.dist_b, self.dist_c, self.dist_d
        return a * (bottom_y ** 3) + b * (bottom_y ** 2) + c * bottom_y + d

    def compute_H(self, h_pixel, D_real):
        return h_pixel * D_real * self.k_H

    def load_calibration(self):
        try:
            with open(CAL_FILE, "r") as f:
                data = ujson.load(f)
            self.dist_a, self.dist_b, self.dist_c, self.dist_d = data["dist_a"], data["dist_b"], data["dist_c"], data["dist_d"]
            if "k_H" in data: self.k_H = data["k_H"]
            return True
        except: return False

    def save_calibration(self):
        data = {"dist_a": self.dist_a, "dist_b": self.dist_b, "dist_c": self.dist_c, "dist_d": self.dist_d, "k_H": self.k_H}
        try:
            with open(CAL_FILE, "w") as f: ujson.dump(data, f)
        except: pass

    def fit_cubic_poly(self):
        n = len(self.cal_lut)
        if n < 4: return False
        A = [[0.0] * 4 for _ in range(4)]
        B = [0.0] * 4
        for yi, di in self.cal_lut:
            row = [yi ** 3, yi ** 2, yi, 1.0]
            for i in range(4):
                for j in range(4): A[i][j] += row[i] * row[j]
                B[i] += row[i] * di
        aug = [A[i] + [B[i]] for i in range(4)]
        for col in range(4):
            max_row = col
            for row in range(col + 1, 4):
                if abs(aug[row][col]) > abs(aug[max_row][col]): max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-15: continue
            for j in range(col, 5): aug[col][j] /= pivot
            for row in range(4):
                if row != col:
                    factor = aug[row][col]
                    for j in range(col, 5): aug[row][j] -= factor * aug[col][j]
        self.dist_a, self.dist_b, self.dist_c, self.dist_d = aug[0][4], aug[1][4], aug[2][4], aug[3][4]
        return True

    def check_uart(self, uart, pl):
        if uart is None: return
        if uart.any():
            try:
                raw_bytes = uart.read()
                if raw_bytes:
                    cmd_str = raw_bytes.decode('utf-8', 'ignore')
                    cmds = cmd_str.split('\n')

                    for cmd in cmds:
                        cmd = cmd.strip()
                        if cmd:
                            self.trigger_popup("UART RX: " + cmd, (255, 255, 0, 0), frames=15)
                            self._handle_command(cmd)
            except: pass

    def trigger_popup(self, text, color, frames=60):
        self.ui_popup_text = text
        self.ui_popup_color = color
        self.ui_popup_frames = frames

    def _do_sample(self):
        idx = self.cal_index
        if idx >= len(self.cal_targets): return
        target_dist = self.cal_targets[idx]
        if len(self.history_bbox) > 0:
            num = len(self.history_bbox)
            s_h = int(sum(b[3] for b in self.history_bbox) / num)
            s_y1 = int(sum(b[1] for b in self.history_bbox) / num)
            bottom_y = s_y1 + s_h
        else: bottom_y = 0
        self.cal_lut.append((float(bottom_y), float(target_dist)))
        self.cal_index = idx + 1
        self.history_bbox.clear()
        self.history_D.clear()
        if self.cal_index >= len(self.cal_targets):
            if self.fit_cubic_poly(): self.save_calibration()
            self.cal_done = True

    def _handle_command(self, cmd):
        valid_cmds = ["CAL_START", "SAMPLE", "CAL_END", "MEAS_START", "RET_IDLE"]
        if cmd not in valid_cmds and not cmd.startswith("K:"):
            return

        if cmd == "CAL_START":
            self.state, self.cal_index, self.cal_lut, self.cal_done = STATE_CALIBRATING, 0, [], False
            self.trigger_popup("SIGNAL 1: CAL_START", (255, 255, 255, 0), frames=60)
            self._do_sample()

        elif cmd == "SAMPLE":
            if self.state == STATE_CALIBRATING and not self.cal_done: self._do_sample()

        elif cmd == "CAL_END":
            if self.state == STATE_CALIBRATING:
                self.state, self.cal_done = STATE_WAIT_MEAS, False
                self.history_bbox.clear(); self.history_D.clear()
                self.trigger_popup("CAL END! WAITING MEASURE...", (255, 0, 255, 0), frames=60)

        elif cmd.startswith("K:"):
            try: self.k_H = float(cmd[2:])
            except: pass

        elif cmd == "MEAS_START":
            self.state, self.meas_buffer_D, self.meas_buffer_H = STATE_MEASURING, [], []
            self.meas_total_frames, self.meas_done = 0, False
            self.history_bbox.clear(); self.history_D.clear()
            self.trigger_popup("MEASURE START! LOCKING...", (255, 0, 255, 255), frames=40)

        elif cmd == "RET_IDLE":
            self.state = STATE_WAIT_MEAS

    def is_bottle_in_center(self, x, y, w, h):
        cx, cy = x + w / 2, y + h / 2
        dw, dh = self.display_size[0], self.display_size[1]
        return (dw / 3 <= cx <= 2 * dw / 3) and (dh / 3 <= cy <= 2 * dh / 3)

    def filter_outliers_mad(self, data):
        n = len(data)
        if n < 3: return data
        sorted_data = sorted(data)
        median_val = sorted_data[n // 2]
        mad = sorted([abs(x - median_val) for x in data])[n // 2]
        if mad < 0.0001: return data
        filtered = [x for x in data if abs(x - median_val) <= 3.0 * mad]
        return filtered if len(filtered) >= 2 else data

    def _draw_state_indicator(self, pl):
        dw, dh = self.display_size[0], self.display_size[1]
        state_names = {STATE_IDLE: "IDLE", STATE_CALIBRATING: "CAL", STATE_WAIT_MEAS: "WAIT", STATE_MEASURING: "MEAS"}
        name = state_names.get(self.state, "???")
        color_map = {STATE_IDLE: (255, 128, 128, 128), STATE_CALIBRATING: (255, 255, 255, 0), STATE_WAIT_MEAS: (255, 0, 255, 0), STATE_MEASURING: (255, 0, 128, 255)}
        pl.osd_img.draw_string_advanced(5, 5, 28, "S:" + name, color=color_map.get(self.state, (255, 128, 128, 128)))

        if self.state == STATE_CALIBRATING:
            if self.cal_done:
                pl.osd_img.draw_string_advanced(dw // 2 - 80, 160, 40, "CAL COMPLETE!", color=(255, 0, 255, 0))
            else:
                if self.cal_index < len(self.cal_targets):
                    pl.osd_img.draw_string_advanced(dw // 2 - 60, 5, 32, "Place @ %d cm" % self.cal_targets[self.cal_index], color=(255, 255, 255, 0))
                pl.osd_img.draw_string_advanced(dw // 2 - 50, 40, 24, "Sampled: %d/%d" % (len(self.cal_lut), len(self.cal_targets)), color=(255, 0, 255, 0))

        if self.ui_popup_frames > 0:
            text_x = max(10, dw // 2 - len(self.ui_popup_text) * 8)
            pl.osd_img.draw_string_advanced(text_x, dh // 2, 35, self.ui_popup_text, color=self.ui_popup_color)
            self.ui_popup_frames -= 1

    def draw_result(self, pl, seg_res, uart):
        self.check_uart(uart, pl)
        pl.osd_img.clear()

        bottle_detected = False

        if seg_res[0]:
            try:
                dets, ids = seg_res[0], seg_res[1]
                best_bottle = None
                max_area = 0
                for i, det in enumerate(dets):
                    class_name = self.labels[int(ids[i])]
                    if class_name in ["bottle", "cup"]:
                        area = det[2] * det[3]
                        if area > max_area:
                            max_area, best_bottle = area, det

                if best_bottle:
                    bottle_detected = True
                    self.miss_count = 0

                    raw_x1, raw_y1, raw_w, raw_h = map(int, best_bottle[:4])
                    self.history_bbox.append([raw_x1, raw_y1, raw_w, raw_h])
                    if len(self.history_bbox) > self.window_size: self.history_bbox.pop(0)

                    num = len(self.history_bbox)
                    s_x1 = int(sum(b[0] for b in self.history_bbox) / num)
                    s_y1 = int(sum(b[1] for b in self.history_bbox) / num)
                    s_w = int(sum(b[2] for b in self.history_bbox) / num)
                    s_h = int(sum(b[3] for b in self.history_bbox) / num)

                    dw, dh = self.display_size[0], self.display_size[1]
                    safe_x1, safe_y1 = max(0, min(s_x1, dw - 1)), max(0, min(s_y1, dh - 1))
                    safe_w, safe_h = max(1, min(s_w, dw - safe_x1)), max(1, min(s_h, dh - safe_y1))

                    bottom_y = safe_y1 + safe_h
                    current_D = self.compute_D(bottom_y)

                    self.history_D.append(current_D)
                    if len(self.history_D) > self.window_size: self.history_D.pop(0)

                    final_D = sum(self.history_D) / len(self.history_D)
                    final_H = self.compute_H(safe_h, final_D)

                    self.last_D, self.last_H = final_D, final_H
                    draw_y = max(0, safe_y1 - 30)

                    pl.osd_img.draw_rectangle(safe_x1, safe_y1, safe_w, safe_h, color=(255, 0, 255, 0), thickness=3)
                    pl.osd_img.draw_circle(safe_x1 + safe_w // 2, bottom_y, 8, color=(255, 255, 0, 0), thickness=2, fill=True)
                    info_text = "D:%.1fcm H:%.1fcm" % (final_D, final_H)
                    pl.osd_img.draw_string_advanced(safe_x1, draw_y, 30, info_text, color=(255, 255, 255, 0))

                    if self.state == STATE_MEASURING:
                        self.meas_total_frames += 1
                        if self.is_bottle_in_center(safe_x1, safe_y1, safe_w, safe_h):
                            self.meas_buffer_D.append(final_D)
                            self.meas_buffer_H.append(final_H)

                            prog_text = "MEAS: %d/%d" % (len(self.meas_buffer_D), self.meas_samples_target)
                            pl.osd_img.draw_string_advanced(safe_x1, max(0, draw_y - 30), 30, prog_text, color=(255, 0, 255, 255))

                            if len(self.meas_buffer_D) >= self.meas_samples_target:
                                fD, fH = self.filter_outliers_mad(self.meas_buffer_D), self.filter_outliers_mad(self.meas_buffer_H)
                                self.last_D, self.last_H = sum(fD) / len(fD), sum(fH) / len(fH)
                                self.meas_done = True
                        else:
                            pl.osd_img.draw_string_advanced(safe_x1, max(0, draw_y - 30), 30, "Move to CENTER!", color=(255, 255, 0, 0))

                        if self.meas_total_frames > 150:
                            self.state, self.meas_done = STATE_WAIT_MEAS, False
                            self.meas_buffer_D.clear(); self.meas_buffer_H.clear()
            except: pass

        if not bottle_detected:
            self.miss_count += 1
            if self.miss_count > 5:
                self.history_bbox.clear()
                self.history_D.clear()
                self.miss_count = 0
            if self.state == STATE_MEASURING: self.meas_total_frames += 1

        self._draw_state_indicator(pl)

    def get_padding_param(self):
        dst_w, dst_h = self.model_input_size[0], self.model_input_size[1]
        ratio_w, ratio_h = float(dst_w) / self.rgb888p_size[0], float(dst_h) / self.rgb888p_size[1]
        ratio = ratio_w if ratio_w < ratio_h else ratio_h
        new_w, new_h = (int)(ratio * self.rgb888p_size[0]), (int)(ratio * self.rgb888p_size[1])
        dw, dh = (dst_w - new_w) / 2, (dst_h - new_h) / 2
        return (int)(round(dh - 0.1)), (int)(round(dh + 0.1)), (int)(round(dw - 0.1)), (int)(round(dw + 0.1))

if __name__ == "__main__":
    display_mode = "lcd"
    display_size = [800, 480] if display_mode == "lcd" else [1920, 1080]

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

    uart = init_uart2()

    try:
        pl = PipeLine(rgb888p_size=[320, 320], display_size=display_size, display_mode=display_mode)
        pl.create(hmirror=True, vflip=True)
        seg = SegmentationApp(kmodel_path, labels=labels, model_input_size=[320, 320],
                              confidence_threshold=0.25, nms_threshold=0.5,
                              mask_threshold=0.5, rgb888p_size=[320, 320],
                              display_size=display_size, debug_mode=0)
        seg.config_preprocess()
        seg.load_calibration()
    except: pass

    frame_count = 0
    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total", 0):
                img = pl.get_frame()
                seg_res = seg.run(img)
                seg.draw_result(pl, seg_res, uart)
                pl.show_image()

                if seg.meas_done:
                    result_str = "D:%.1f,H:%.1f\n" % (seg.last_D, seg.last_H)
                    if uart: uart.write(result_str.encode('utf-8'))
                    seg.meas_done = False
                    seg.state = STATE_WAIT_MEAS

                frame_count += 1
                if frame_count % 30 == 0: gc.collect()

                # 【量产版核心 2】强制降低计算频率！
                # 脱机供电不稳定时，让 CPU 稍微喘口气，绝不能死命跑！
                time.sleep_ms(30)

    except: pass
    finally:
        try:
            seg.deinit()
            pl.destroy()
        except: pass
