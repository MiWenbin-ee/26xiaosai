import time, os, sys, gc
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
from libs.AIBase import AIBase
from libs.AI2D import Ai2d

class YOLOv8App(AIBase):
    def __init__(self, kmodel_path, labels, model_size, sensor_size):
        super().__init__(kmodel_path, model_input_size=model_size, rgb888p_size=sensor_size)
        self.labels = labels
        self.model_size = model_size
        self.sensor_size = sensor_size
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self):
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([1, 3, self.sensor_size[1], self.sensor_size[0]],
                        [1, 3, self.model_size[1], self.model_size[0]])

    def postprocess(self, results):
        # 修复点 2: 将宽高参数修正为标准的 [Width, Height] 顺序
        return aidemo.yolov8_det_postprocess(
            results,
            [self.sensor_size[0], self.sensor_size[1]],
            self.model_size,
            [self.sensor_size[0], self.sensor_size[1]],
            len(self.labels), 0.35, 0.5, 30
        )

if __name__ == "__main__":
    os.exitpoint()

    # 修复点 3: 显式分配 VB 内存池，防止内存溢出死机
    config = k_vb_config()
    config.max_poolcnt = 1
    # 预留足够的内存给 640x480 显示 以及 320x320 的 AI 缩放处理
    config.pool_config[0].blk_size = 640 * 480 * 3 + 320 * 320 * 3
    config.pool_config[0].blk_cnt = 3
    config.pool_config[0].mode = VB_REMAP_MODE.NOCACHE
    media.buffer_config(config)

    MediaManager.init()

    W_DISP, H_DISP = 640, 480

    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    app = YOLOv8App("/sdcard/yolov8n_320.kmodel", labels, [320, 320], [W_DISP, H_DISP])
    app.config_preprocess()

    sensor = Sensor()
    sensor.reset()

    # 修复点 1: 注释掉未使用的 chn=0 (1080P) 配置，极大节省内存
    # sensor.set_framesize(width=1920, height=1080, chn=0)
    # sensor.set_pixformat(Sensor.RGB888, chn=0)

    sensor.set_framesize(width=W_DISP, height=H_DISP, chn=1)
    sensor.set_pixformat(Sensor.RGB888, chn=1)

    Display.init(Display.VIRT, width=W_DISP, height=H_DISP, fps=30)

    sensor.run()

    try:
        while True:
            os.exitpoint()
            img = sensor.snapshot(chn=1)
            dets = app.run(img)

            if dets:
                for det in dets:
                    cls_id = int(det[5])
                    if cls_id == 39: # 39 是瓶子的类 ID
                        x, y, w, h = map(int, det[:4])
                        score = det[4]
                        img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=3)

                        # 修复点 4: 使用基础 draw_string 并加入 max(0, y-35) 保护，防止文字画出屏幕外引起崩溃
                        safe_y = max(0, y - 35)
                        img.draw_string(x, safe_y, f"Bottle {int(score*100)}%", color=(0, 255, 0), scale=2)

            Display.show_image(img)
            gc.collect() # 显式回收内存非常重要

    except Exception as e:
        print("\n❌ 发生异常:", repr(e))
    finally:
        app.deinit()
        sensor.stop()
        Display.deinit()
        MediaManager.deinit()
