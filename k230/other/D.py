import time
import image
from media.sensor import *
from media.display import *
from media.media import media

# ==========================================
# 1. 标定参数与全局变量配置
# ==========================================
CALIB_TARGETS = [50, 60, 70, 80, 90, 100]  # 标定距离序列
calib_y_vals = [0] * len(CALIB_TARGETS)    # 用于存放记录的 Y 坐标
calib_step = 0                             # 当前标定到了第几个点
is_calibrated = False                      # 是否完成标定

countdown = 5                              # 倒计时秒数
last_time = 0                              # 时间戳记录

# 【极其重要】：请修改为你测试物体的 HSV 颜色阈值！
# (L_min, L_max, A_min, A_max, B_min, B_max) 或者 (H_min, H_max, S_min, S_max, V_min, V_max)
TARGET_COLOR_THRESHOLD = (30, 100, 15, 127, 15, 127)

# ==========================================
# 2. 分段线性插值测距算法
# ==========================================
def calculate_distance(current_y):
    if not is_calibrated:
        return 0.0

    # 处理超出标定范围的情况 (通常距离越近，Y坐标越大/越靠画面底部)
    # 假设 calib_y_vals[0] 是 50cm 的 Y (最大)，calib_y_vals[-1] 是 100cm 的 Y (最小)
    if current_y >= max(calib_y_vals[0], calib_y_vals[1]):
        return CALIB_TARGETS[0]
    if current_y <= min(calib_y_vals[-1], calib_y_vals[-2]):
        return CALIB_TARGETS[-1]

    # 遍历寻找 current_y 落在哪两个标定点之间
    for i in range(len(calib_y_vals) - 1):
        y1 = calib_y_vals[i]
        y2 = calib_y_vals[i+1]

        # 检查当前 Y 是否在这两个标定点之间
        if min(y1, y2) <= current_y <= max(y1, y2):
            d1 = CALIB_TARGETS[i]
            d2 = CALIB_TARGETS[i+1]

            # 直线比例插值计算
            ratio = (current_y - y1) / (y2 - y1)
            distance = d1 + ratio * (d2 - d1)
            return distance

    return 0.0

# ==========================================
# 3. 硬件初始化
# ==========================================
media.init()
Display.init(Display.VIRT, width=640, height=480, fps=30)
sensor = Sensor()
sensor.reset()
sensor.set_framesize(width=640, height=480)
sensor.set_pixformat(Sensor.RGB565)
sensor.run()

print("系统就绪，请看屏幕指示...")
clock = time.clock()
last_time = time.ticks_ms()

# ==========================================
# 4. 主循环
# ==========================================
try:
    while True:
        clock.tick()
        img = sensor.snapshot()

        # 寻找目标色块
        blobs = img.find_blobs([TARGET_COLOR_THRESHOLD], pixels_threshold=150, area_threshold=150)
        cup_bottom_y = -1 # 默认没找到

        if blobs:
            # 找到最大的色块
            largest_blob = max(blobs, key=lambda b: b.pixels())
            img.draw_rectangle(largest_blob.rect(), color=(255, 0, 0), thickness=2)

            # 计算杯底 Y 坐标 (框的顶部 y + 框的高度 h)
            cup_bottom_y = largest_blob.y() + largest_blob.h()

            # 画一个绿色十字标记杯底
            img.draw_cross(largest_blob.cx(), cup_bottom_y, color=(0, 255, 0), size=10, thickness=2)

        # ----------------------------------------
        # 状态机：标定模式 vs 测距模式
        # ----------------------------------------
        if not is_calibrated:
            # 屏幕左上角提示当前需要放的距离
            msg = "Calibrating: {} cm".format(CALIB_TARGETS[calib_step])
            img.draw_string(10, 10, msg, scale=2, color=(255, 255, 0))

            if cup_bottom_y != -1:
                # 画面里有杯子，开始倒计时
                current_time = time.ticks_ms()
                if time.ticks_diff(current_time, last_time) >= 1000: # 满 1 秒
                    countdown -= 1
                    last_time = current_time

                img.draw_string(10, 40, "Hold still... {}s".format(countdown), scale=2, color=(255, 0, 0))

                # 倒计时结束，记录数据！
                if countdown <= 0:
                    calib_y_vals[calib_step] = cup_bottom_y
                    print("-> 已记录 {}cm, Y坐标: {}".format(CALIB_TARGETS[calib_step], cup_bottom_y))

                    calib_step += 1
                    countdown = 5 # 重置倒计时，准备下一个点

                    # 检查是否全部标定完成
                    if calib_step >= len(CALIB_TARGETS):
                        is_calibrated = True
                        print("标定全部完成！插值表:", calib_y_vals)
            else:
                # 画面里没杯子，倒计时暂停并重置时间戳
                img.draw_string(10, 40, "Looking for object...", scale=2, color=(255, 165, 0))
                last_time = time.ticks_ms()

        else:
            # 已经标定完了，实时显示物理距离！
            if cup_bottom_y != -1:
                dist = calculate_distance(cup_bottom_y)
                img.draw_string(10, 10, "Distance: {:.1f} cm".format(dist), scale=3, color=(0, 255, 0))
            else:
                img.draw_string(10, 10, "Distance: -- cm", scale=3, color=(255, 0, 0))

        Display.show_image(img)

except KeyboardInterrupt:
    print("停止运行")
finally:
    sensor.stop()
    Display.deinit()
    media.deinit()
    print("已释放资源")
