import time, os, gc, sys
import image

# K230 全新的核心多媒体库
from media.sensor import *
from media.display import *
from media.media import *

try:
    print("K230 多媒体引擎初始化中...")
    # 初始化虚拟显示屏 (VIRT)，专用于通过数据线将画面回传到 IDE 右侧
    Display.init(Display.VIRT, width=640, height=480, fps=30, to_ide=True)

    # 启动媒体大管家
    MediaManager.init()

    print("初始化摄像头...")
    # K230 板载摄像头的接口通常是 CSI2，所以默认 id=2 (如果不亮可以改成0或1试试)
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_framesize(width=640, height=480) # 设为 VGA 分辨率
    sensor.set_pixformat(Sensor.RGB565)

    # 正式启动传感器传输
    sensor.run()

    print("系统运行正常！请看 IDE 右侧图像缓冲区。")

    # 用于手动计算 FPS 的时间戳
    last_time = time.ticks_ms()

    while True:
        # 检测 IDE 中是否点击了红色的停止按钮，这句在 K230 中非常重要！
        os.exitpoint()

        # 1. 抓取一帧画面
        img = sensor.snapshot()

        # 2. 画十字准星 (X:320, Y:240 是 640x480 的中心点)
        img.draw_cross(320, 240, color=(255, 0, 0), size=15, thickness=2)

        # 3. 计算 FPS
        current_time = time.ticks_ms()
        dt = time.ticks_diff(current_time, last_time)
        last_time = current_time
        fps = 1000.0 / dt if dt > 0 else 0

        # 4. 打印文字：K230 推荐使用 draw_string_advanced，30 是字体大小
        img.draw_string_advanced(10, 10, 30, f"FPS: {fps:.1f}", color=(0, 255, 0))

        # 5. 必须显式调用显示指令，画面才会回传到 IDE
        Display.show_image(img)

except KeyboardInterrupt:
    print("用户手动停止了程序")
except Exception as e:
    print(f"程序异常退出: {e}")

finally:
    # ！！！K230 核心避坑指南！！！
    # 无论程序是正常结束、还是报错崩溃，用完必须手动释放底层的多媒体硬件资源。
    # 否则你再次点击运行会报 "Device Busy" 或者干脆死机重启。
    print("正在释放硬件资源，请稍候...")
    if 'sensor' in locals() and sensor is not None:
        sensor.stop()
    Display.deinit()
    MediaManager.deinit()
    gc.collect()
    print("资源释放完毕，可安全再次运行！")
