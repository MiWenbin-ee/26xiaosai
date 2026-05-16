import time
time.sleep(2) # 开机等待电压平稳

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
        # 针对 01Studio 板子：排针 Pin11(IO5), Pin12(IO6)
        fpioa.set_function(5, FPIOA.UART2_TXD)
        fpioa.set_function(6, FPIOA.UART2_RXD)
        # 启用独立干净的 UART2，防止与电脑 IDE 冲突
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
        self.k_H = 0.00091 # 初始高度常数
        self.k_W = 0.00091 # 初始宽度常数

        self.REAL_BOTTLE_HEIGHT = 17.0
        self.REAL_BOTTLE_WIDTH = 6.5

        self.state = STATE_IDLE
        self.meas_buffer_D = []
        self.meas_buffer_H = []
        self.meas_buffer_L = []
        self.meas_buffer_C = [] # 颜色缓冲
        self.meas_samples_target = 10
        self.meas_total_frames = 0
        self.meas_done = False

        self.last_D, self.last_H, self.last_L, self.last_C = 0.0, 0.0, 0.0, 0

        self.window_size = 20
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
        return a * (bottom_y ** 3) + b * (bottom_y ** 2) + c * bottom_y + d - self.REAL_BOTTLE_WIDTH/2.0

    def compute_H(self, h_pixel, D_real):
        return h_pixel * D_real * self.k_H

    def get_frequent_lowest_L(self, data):
        if not data: return 0.0
        freq_map = {}
        for v in data:
            bin_key = round(v * 2) / 2.0
            freq_map[bin_key] = freq_map.get(bin_key, 0) + 1

        max_freq = max(freq_map.values())
        candidates = [k for k, v in freq_map.items() if v == max_freq]
        best_bin = min(candidates)

        exact_vals = [v for v in data if round(v * 2) / 2.0 == best_bin]
        return sum(exact_vals) / len(exact_vals)

    # ========================================================
    # 多列 ROI 投票扫描机制（抵抗图案/磨砂）
    # ========================================================
    def measure_liquid_level(self, img, x, y, w, h):
        bottom_y = y + h
        if h < 5: return bottom_y

        scan_start = y + int(h * 0.85)
        scan_end = y + int(h * 0.15)

        img_w = self.rgb888p_size[0]
        img_h = self.rgb888p_size[1]
        scale_x = img_w / float(self.display_size[0])
        scale_y = img_h / float(self.display_size[1])

        columns_x = [x + int(w * ratio) for ratio in [0.2, 0.35, 0.5, 0.65, 0.8]]
        liquid_ys = []

        for cx in columns_x:
            col_max_diff = 0.0
            col_liquid_y = bottom_y

            def get_luma(cy):
                luma_sum = 0
                valid = 0
                img_cy = int(cy * scale_y)
                for dx in [-2, 0, 2]:
                    img_cx = int((cx + dx) * scale_x)
                    if 0 <= img_cx < img_w and 0 <= img_cy < img_h:
                        try:
                            if len(img.shape) == 3:
                                if img.shape[0] == 3:
                                    r, g, b = img[0, img_cy, img_cx], img[1, img_cy, img_cx], img[2, img_cy, img_cx]
                                else:
                                    r, g, b = img[img_cy, img_cx, 0], img[img_cy, img_cx, 1], img[img_cy, img_cx, 2]
                            elif len(img.shape) == 1:
                                area = img_w * img_h
                                idx = img_cy * img_w + img_cx
                                r, g, b = img[idx], img[area + idx], img[2 * area + idx]
                            else: continue
                            luma_sum += (float(r) * 0.299 + float(g) * 0.587 + float(b) * 0.114)
                            valid += 1
                        except: pass
                return luma_sum / valid if valid > 0 else -1.0

            prev_luma = get_luma(scan_start)
            if prev_luma < 0:
                liquid_ys.append(bottom_y)
                continue

            for row in range(scan_start - 1, scan_end, -2):
                curr_luma = get_luma(row)
                if curr_luma < 0: continue

                diff = abs(curr_luma - prev_luma)
                if diff > col_max_diff and diff > 4.0:
                    col_max_diff = diff
                    col_liquid_y = row
                prev_luma = curr_luma

            liquid_ys.append(col_liquid_y)

        liquid_ys.sort()
        final_liquid_y = liquid_ys[len(liquid_ys) // 2]
        return final_liquid_y

    # ========================================================
    # 颜色 HSV 空间识别，核心区域统计投票
    # ========================================================
    def recognize_color(self, img, x, y, w, h, liquid_y):
        bottom_y = y + h
        if bottom_y - liquid_y < 10: return 0

        core_top = liquid_y + int((bottom_y - liquid_y) * 0.3)
        core_bottom = bottom_y - int((bottom_y - liquid_y) * 0.2)
        core_left = x + int(w * 0.3)
        core_right = x + int(w * 0.7)

        if core_bottom <= core_top or core_right <= core_left: return 0

        img_w, img_h = self.rgb888p_size[0], self.rgb888p_size[1]
        scale_x, scale_y = img_w / float(self.display_size[0]), img_h / float(self.display_size[1])

        color_votes = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
        step_y = max(1, (core_bottom - core_top) // 5)
        step_x = max(1, (core_right - core_left) // 5)

        for cy in range(core_top, core_bottom, step_y):
            for cx in range(core_left, core_right, step_x):
                img_cy, img_cx = int(cy * scale_y), int(cx * scale_x)
                if 0 <= img_cx < img_w and 0 <= img_cy < img_h:
                    try:
                        if len(img.shape) == 3:
                            if img.shape[0] == 3:
                                r, g, b = img[0, img_cy, img_cx], img[1, img_cy, img_cx], img[2, img_cy, img_cx]
                            else:
                                r, g, b = img[img_cy, img_cx, 0], img[img_cy, img_cx, 1], img[img_cy, img_cx, 2]
                        elif len(img.shape) == 1:
                            area = img_w * img_h
                            idx = img_cy * img_w + img_cx
                            r, g, b = img[idx], img[area + idx], img[2 * area + idx]
                        else: continue

                        r_f, g_f, b_f = float(r)/255.0, float(g)/255.0, float(b)/255.0
                        mx, mn = max(r_f, g_f, b_f), min(r_f, g_f, b_f)
                        df = mx - mn
                        h_val = 0
                        if mx == mn: h_val = 0
                        elif mx == r_f: h_val = (60 * ((g_f - b_f) / df) + 360) % 360
                        elif mx == g_f: h_val = (60 * ((b_f - r_f) / df) + 120) % 360
                        elif mx == b_f: h_val = (60 * ((r_f - g_f) / df) + 240) % 360

                        s_val = 0 if mx == 0 else (df / mx) * 100
                        v_val = mx * 100

                        cid = 0
                        if v_val < 25: cid = 5
                        elif s_val < 25 and v_val > 60: cid = 6
                        elif s_val < 30: cid = 0
                        else:
                            if h_val <= 15 or h_val >= 330: cid = 1
                            elif 15 < h_val <= 75: cid = 4
                            elif 75 < h_val <= 165: cid = 2
                            elif 165 < h_val <= 270: cid = 3
                            elif 270 < h_val < 330: cid = 7

                        color_votes[cid] += 1
                    except: pass

        return max(color_votes, key=color_votes.get)

    def load_calibration(self):
        try:
            with open(CAL_FILE, "r") as f:
                data = ujson.load(f)
            self.dist_a, self.dist_b, self.dist_c, self.dist_d = data["dist_a"], data["dist_b"], data["dist_c"], data["dist_d"]
            if "k_H" in data: self.k_H = data["k_H"]
            if "k_W" in data: self.k_W = data["k_W"]
            return True
        except: return False

    def save_calibration(self):
        data = {"dist_a": self.dist_a, "dist_b": self.dist_b, "dist_c": self.dist_c, "dist_d": self.dist_d, "k_H": self.k_H, "k_W": self.k_W}
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
            s_w = int(sum(b[2] for b in self.history_bbox) / num)
            s_y1 = int(sum(b[1] for b in self.history_bbox) / num)
            bottom_y = s_y1 + s_h
        else:
            bottom_y = 0
            s_h = 1
            s_w = 1

        if self.cal_index == 0 and s_h > 0 and s_w > 0:
            self.k_H = self.REAL_BOTTLE_HEIGHT / (float(s_h) * 50.0)
            self.k_W = self.REAL_BOTTLE_WIDTH / (float(s_w) * 50.0)
            self.trigger_popup("K_H & K_W CALIBRATED!", (0, 255, 0, 255), frames=40)

        self.cal_lut.append((float(bottom_y), float(target_dist)))
        self.cal_index = idx + 1
        self.history_bbox.clear()
        self.history_D.clear()
        if self.cal_index >= len(self.cal_targets):
            if self.fit_cubic_poly(): self.save_calibration()
            self.cal_done = True

    def _handle_command(self, cmd):
        valid_cmds = ["CAL_START", "SAMPLE", "CAL_END", "MEAS_START", "RET_IDLE"]
        if cmd not in valid_cmds and not cmd.startswith("K:"): return

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
            self.state, self.meas_buffer_D, self.meas_buffer_H, self.meas_buffer_L, self.meas_buffer_C = STATE_MEASURING, [], [], [], []
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

    def draw_result(self, pl, img, seg_res, uart):
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
                    if class_name in ["bottle", "cup", "wine glass", "vase", "bowl", "glass"]:
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

# 1. 直接算出中心距离 (因为你们标定就是对准中心的，多项式自带外挂)
                    raw_D = self.compute_D(bottom_y)

                    # 2. 对中心距离进行滑动平均滤波，防止数值跳动
                    self.history_D.append(raw_D)
                    if len(self.history_D) > self.window_size: self.history_D.pop(0)
                    final_D = sum(self.history_D) / len(self.history_D)+2

                    # 3. 使用平滑后的绝对中心距离 (final_D)，科学计算高度和宽度！
                    final_H = self.compute_H(safe_h, final_D)
                    physical_width = safe_w * final_D * self.k_W

                    pl.osd_img.draw_rectangle(safe_x1, safe_y1, safe_w, safe_h, color=(255, 0, 255, 0), thickness=3)
                    draw_y = max(0, safe_y1 - 30)

                    liquid_y = bottom_y
                    final_L = 0.0
                    color_id = 0

                    try:
                        liquid_y = self.measure_liquid_level(img, safe_x1, safe_y1, safe_w, safe_h)
                        liquid_ratio = ((safe_y1 + safe_h) - liquid_y) / float(safe_h) if safe_h > 0 else 0
                        final_L = final_H * liquid_ratio

                        color_id = self.recognize_color(img, safe_x1, safe_y1, safe_w, safe_h, liquid_y)

                        pl.osd_img.draw_line(safe_x1, liquid_y, safe_x1 + safe_w, liquid_y, color=(255, 0, 0, 255), thickness=3)
                        pl.osd_img.draw_circle(safe_x1 + safe_w // 2, bottom_y, 8, color=(255, 255, 0, 0), thickness=2, fill=True)
                    except Exception as e:
                        print("Liquid Level Scan Error:", e)

                    COLOR_NAMES = {0:"NONE", 1:"RED", 2:"GREEN", 3:"BLUE", 4:"YELLOW", 5:"BLACK", 6:"WHITE", 7:"PURPLE"}
                    color_str = COLOR_NAMES.get(color_id, "UNK")

                    self.last_D, self.last_H, self.last_L, self.last_C = final_D, final_H, final_L, color_id
                    info_text = "D:%.1f H:%.1f L:%.1f W:%.1f C:%s" % (final_D, final_H, final_L, physical_width, color_str)
                    pl.osd_img.draw_string_advanced(safe_x1, draw_y, 30, info_text, color=(255, 255, 255, 0))

                    if self.state == STATE_MEASURING:
                        self.meas_total_frames += 1
                        if self.is_bottle_in_center(safe_x1, safe_y1, safe_w, safe_h):
                            self.meas_buffer_D.append(final_D)
                            self.meas_buffer_H.append(final_H)
                            self.meas_buffer_L.append(final_L)
                            self.meas_buffer_C.append(color_id)

                            prog_text = "MEAS: %d/%d" % (len(self.meas_buffer_D), self.meas_samples_target)
                            pl.osd_img.draw_string_advanced(safe_x1, max(0, draw_y - 60), 30, prog_text, color=(255, 0, 255, 255))

                            if len(self.meas_buffer_D) >= self.meas_samples_target:
                                fD = self.filter_outliers_mad(self.meas_buffer_D)
                                fH = self.filter_outliers_mad(self.meas_buffer_H)

                                self.last_D = sum(fD) / len(fD) if len(fD) > 0 else final_D
                                self.last_H = sum(fH) / len(fH) if len(fH) > 0 else final_H
                                self.last_L = self.get_frequent_lowest_L(self.meas_buffer_L)
                                self.last_C = max(set(self.meas_buffer_C), key=self.meas_buffer_C.count)

                                self.meas_done = True
                        else:
                            pl.osd_img.draw_string_advanced(safe_x1, max(0, draw_y - 60), 30, "Move to CENTER!", color=(255, 255, 0, 0))

                        if self.meas_total_frames > 150:
                            self.state, self.meas_done = STATE_WAIT_MEAS, False
                            self.meas_buffer_D.clear(); self.meas_buffer_H.clear()
                            self.meas_buffer_L.clear(); self.meas_buffer_C.clear()
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
        # 【画面端正化】关闭了水平镜像和垂直翻转
        pl.create(hmirror=False, vflip=False)
        seg = SegmentationApp(kmodel_path, labels=labels, model_input_size=[320, 320],
                              confidence_threshold=0.15, nms_threshold=0.5,
                              mask_threshold=0.5, rgb888p_size=[320, 320],
                              display_size=display_size, debug_mode=0)
        seg.config_preprocess()
        seg.load_calibration()
    except: pass

    frame_count = 0
    last_seg_res = [[], [], []]

    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total", 0):
                img = pl.get_frame()

                if frame_count % 2 == 0:
                    last_seg_res = seg.run(img)

                seg.draw_result(pl, img, last_seg_res, uart)

                if seg.meas_done:
                    result_str = "D:%.1f,H:%.1f,L:%.1f,C:%d\n" % (seg.last_D, seg.last_H, seg.last_L, seg.last_C)
                    if uart: uart.write(result_str.encode('utf-8'))
                    seg.meas_done = False
                    seg.state = STATE_WAIT_MEAS

                pl.show_image()

                frame_count += 1
                if frame_count % 20 == 0: gc.collect()
                time.sleep_ms(10)

    except: pass
    finally:
        try:
            seg.deinit()
            pl.destroy()
        except: pass
