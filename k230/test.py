import time, os, gc, sys
from media.sensor import *
from media.display import *
from media.media import *

try:
    print("开始配置外设参数...")

    # 1. 配置摄像头 (保持不变)
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=1920, height=1080, chn=0)
    sensor.set_pixformat(Sensor.RGB888, chn=0)

    # 2. 配置物理屏幕的分辨率 (根据你的屏幕大小，通常是 800x480)
    W_DISP = 800
    H_DISP = 480

    # 让缩放通道输出和物理屏幕一样大的分辨率
    sensor.set_framesize(width=W_DISP, height=H_DISP, chn=1)
    sensor.set_pixformat(Sensor.RGB888, chn=1)

    # 3. 【核心修改点】驱动物理屏幕！
    # 使用 ST7701 驱动 MIPI 屏幕，to_ide=True 让电脑端也同步显示
    print("唤醒物理 MIPI 屏幕...")
    Display.init(Display.ST7701, width=W_DISP, height=H_DISP, to_ide=True)

    print("K230 多媒体引擎初始化并分配内存...")
    MediaManager.init()

    print("启动传感器传输...")
    sensor.run()

    while True:
        os.exitpoint()

        # 从 chn=1 抓取画面
        img = sensor.snapshot(chn=1)

        if img is not None:
            # 画一个红色的十字准星 (画在屏幕正中间)
            img.draw_cross(int(W_DISP/2), int(H_DISP/2), color=(255, 0, 0), size=20, thickness=3)

            # 推送到屏幕显示
            Display.show_image(img)
        else:
            time.sleep(0.1)

except Exception as e:
    print(f"程序异常退出: {repr(e)}")
finally:
    if 'sensor' in locals() and sensor is not None:
        sensor.stop()
    Display.deinit()
    MediaManager.deinit()
    gc.collect()
    print("资源释放完毕。")
