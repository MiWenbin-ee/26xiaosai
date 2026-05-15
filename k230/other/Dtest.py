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

# 自定义YOLOv8分割类
class SegmentationApp(AIBase):
    def __init__(self,kmodel_path,labels,model_input_size,confidence_threshold=0.2,nms_threshold=0.5,mask_threshold=0.5,rgb888p_size=[224,224],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.labels=labels
        self.model_input_size=model_input_size
        self.confidence_threshold=confidence_threshold
        self.nms_threshold=nms_threshold
        self.mask_threshold=mask_threshold
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.color_four=[(255, 220, 20, 60), (255, 119, 11, 32), (255, 0, 0, 142), (255, 0, 0, 230),
                         (255, 106, 0, 228), (255, 0, 60, 100), (255, 0, 80, 100), (255, 0, 0, 70),
                         (255, 0, 0, 192), (255, 250, 170, 30), (255, 100, 170, 30), (255, 220, 220, 0),
                         (255, 175, 116, 175), (255, 250, 0, 30), (255, 165, 42, 42), (255, 255, 77, 255),
                         (255, 0, 226, 252), (255, 182, 182, 255), (255, 0, 82, 0), (255, 120, 166, 157)]
        self.masks=np.zeros((1,self.display_size[1],self.display_size[0],4))
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

        # ================== 改进版滤波状态变量 ==================
        self.is_first_frame = True
        self.smoothed_bbox = [0.0, 0.0, 0.0, 0.0]  # 用于存储平滑后的 [x1, y1, w, h]
        # ========================================================

    def config_preprocess(self,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            top,bottom,left,right=self.get_padding_param()
            self.ai2d.pad([0,0,0,0,top,bottom,left,right], 0, [114,114,114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            seg_res = aidemo.segment_postprocess(results,[self.rgb888p_size[1],self.rgb888p_size[0]],self.model_input_size,[self.display_size[1],self.display_size[0]],self.confidence_threshold,self.nms_threshold,self.mask_threshold,self.masks)
            return seg_res

    def draw_result(self,pl,seg_res):
        with ScopedTiming("display_draw",self.debug_mode >0):
            if seg_res[0]:
                pl.osd_img.clear()
                mask_img=image.Image(self.display_size[0], self.display_size[1], image.ARGB8888,alloc=image.ALLOC_REF,data=self.masks)
                pl.osd_img.copy_from(mask_img)
                dets,ids,scores = seg_res[0],seg_res[1],seg_res[2]

                bottle_detected = False

                for i, det in enumerate(dets):
                    class_id = int(ids[i])
                    class_name = self.labels[class_id]

                    # 提取原始浮点坐标
                    raw_x1, raw_y1, raw_w, raw_h = det[0], det[1], det[2], det[3]

                    if class_name == "bottle":
                        bottle_detected = True

                        # --- 核心改进：对几何坐标进行滤波 ---
                        alpha_box = 0.25 # 坐标滤波系数：0.25 能提供极强的稳定性，如果觉得拖影重可调到 0.4

                        if self.is_first_frame:
                            self.smoothed_bbox = [raw_x1, raw_y1, raw_w, raw_h]
                            self.is_first_frame = False
                        else:
                            self.smoothed_bbox[0] = alpha_box * raw_x1 + (1 - alpha_box) * self.smoothed_bbox[0]
                            self.smoothed_bbox[1] = alpha_box * raw_y1 + (1 - alpha_box) * self.smoothed_bbox[1]
                            self.smoothed_bbox[2] = alpha_box * raw_w + (1 - alpha_box) * self.smoothed_bbox[2]
                            self.smoothed_bbox[3] = alpha_box * raw_h + (1 - alpha_box) * self.smoothed_bbox[3]

                        # 取平滑后的坐标进行绘制和计算
                        s_x1, s_y1, s_w, s_h = map(int, self.smoothed_bbox)

                        # 绘制稳定的检测框
                        pl.osd_img.draw_rectangle(s_x1, s_y1, s_w, s_h, color=(255, 0, 255, 0), thickness=2)
                        pl.osd_img.draw_string_advanced(s_x1, s_y1 - 40, 32, " " + class_name + " " + str(round(scores[i],2)), color=self.get_color(class_id))

                        # 1. 提取稳定的底部坐标
                        bottom_x = int(s_x1 + s_w / 2)
                        bottom_y = int(s_y1 + s_h)

                        # 2. 画准星标记 (现在它会稳稳吸附在杯底)
                        pl.osd_img.draw_circle(bottom_x, bottom_y, 8, color=(255, 255, 0, 0), thickness=2, fill=True)

                        # 3. 填入你的二阶拟合系数 (针对远端较准的参数)
                        a = -0.00000873
                        b = 0.01196712
                        c = -5.65386753
                        d = 964.76871412

                        # 4. 计算距离D (由于 y 已经是平滑的，D 也会非常平滑)
                        final_D = a * (bottom_y ** 3) + b * (bottom_y ** 2) + c * bottom_y + d

                        # 5. 屏幕显示
                        display_text = f"D: {final_D:.1f}cm (Y:{bottom_y})"
                        pl.osd_img.draw_string_advanced(s_x1, s_y1 + s_h + 10, 35, display_text, color=(255, 255, 0, 0))

                    else:
                        # 对于非 bottle 的物体，依然使用原始抖动坐标绘制，节省算力
                        x1, y1, w, h = map(int, det)
                        pl.osd_img.draw_string_advanced(x1, y1-40, 32, " " + class_name + " " + str(round(scores[i],2)), color=self.get_color(class_id))

                # 如果这帧完全没检测到瓶子，重置状态，防止目标消失后重新出现时带着错误的残影
                if not bottle_detected:
                    self.is_first_frame = True

            else:
                pl.osd_img.clear()
                self.is_first_frame = True

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
        return  top, bottom, left, right

    def get_color(self, x):
        idx=x%len(self.color_four)
        return self.color_four[idx]


if __name__=="__main__":
    display_mode="lcd"
    if display_mode=="hdmi":
        display_size=[1920,1080]
    else:
        display_size=[800,480]

    kmodel_path="/sdcard/app/yolov8n_320.kmodel"
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    confidence_threshold = 0.2
    nms_threshold = 0.5
    mask_threshold=0.5
    rgb888p_size=[320,320]

    pl=PipeLine(rgb888p_size=rgb888p_size,display_size=display_size,display_mode=display_mode)
    pl.create(hmirror=True,vflip=True)
    seg=SegmentationApp(kmodel_path,labels=labels,model_input_size=[320,320],confidence_threshold=confidence_threshold,nms_threshold=nms_threshold,mask_threshold=mask_threshold,rgb888p_size=rgb888p_size,display_size=display_size,debug_mode=0)
    seg.config_preprocess()
    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total",1):
                img=pl.get_frame()
                seg_res=seg.run(img)
                seg.draw_result(pl,seg_res)
                pl.show_image()
                gc.collect()
    except Exception as e:
        sys.print_exception(e)
    finally:
        seg.deinit()
        pl.destroy()
