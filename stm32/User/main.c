#include "stm32f10x.h"
#include "Delay.h"
#include "OLED.h"
#include "ADC.h"
#include "Key.h"
#include "LED.h"
#include "UART.h"
#include <stdio.h>
#include <string.h>

enum {
    STATE_IDLE,
    STATE_CALIBRATING,
    STATE_WAIT_MEAS,
    STATE_MEASURING,
    STATE_DISPLAY
};

// ========================================================
// 电流 2 秒滑动平均滤波算法 (主循环100ms x 20次 = 2秒)
// ========================================================
#define CURRENT_SAMPLES 20
float current_buf[CURRENT_SAMPLES] = {0};
uint8_t current_idx = 0;
uint8_t current_filled = 0;

float Get_Averaged_Current(void)
{
    // 获取瞬时电流 (假设你的 ADC.h 里有这个函数)
    float raw_c = Get_Current_Amps(); 
    
    // 装入环形缓冲区
    current_buf[current_idx] = raw_c;
    current_idx = (current_idx + 1) % CURRENT_SAMPLES;
    if (current_filled < CURRENT_SAMPLES) current_filled++;

    // 计算这2秒内的平均值
    float sum = 0;
    for (int i = 0; i < current_filled; i++) {
        sum += current_buf[i];
    }
    return sum / current_filled;
}

// 刷新第一行：实时动态显示 2 秒平滑后的电流与功率 (保留2位小数)
static float OLED_ShowPowerLine(void)
{
    char buf[17];
    float avg_current = Get_Averaged_Current();
    float avg_power = 5.0f * avg_current;

    // 1. 电流逻辑：整体放大100倍并四舍五入，保留2位小数
    int i_total = (int)(avg_current * 100.0f + 0.5f);
    int i_w = i_total / 100;    // 电流整数部分
    int i_dw = i_total % 100;   // 电流两位小数部分

    // 2. 功率逻辑：整体放大100倍并四舍五入，保留2位小数
    int p_total = (int)(avg_power * 100.0f + 0.5f);
    int p_w = p_total / 100;    // 功率整数部分
    int p_dw = p_total % 100;   // 功率两位小数部分

    // 3. 格式化打印：使用 %02d 保证个位数小数时补零
    int len = sprintf(buf, "I:%d.%02dA P:%d.%02dW",
                    i_w, i_dw,
                    p_w, p_dw);
                    
    // 4. 严格限制长度，防止 OLED 刷屏错位
    while (len < 16) buf[len++] = ' ';
    buf[16] = '\0';
    OLED_ShowString(1, 1, buf);
    
    return avg_power;
}

int main(void)
{
    uint8_t state = STATE_IDLE;
    uint8_t key;
    char rx_buf[64];
    char display_buf[24];
    
    float last_D = 0.0f, last_H = 0.0f, last_L = 0.0f;
    int last_C = 0;
    float max_power = 0.0f;
    
    uint16_t display_ticks = 0;
    uint16_t meas_timeout = 0;
    uint8_t cal_sample_count = 0;

    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);

    OLED_Init();
    LED_Init();
    Key_Init();
    ADCDMA_Init();
    UART1_Init(115200);

    LED1_OFF();
    LED2_OFF();

    // 开机初始界面 (只管 2 3 4 行，第 1 行由死循环接管)
    OLED_ShowString(2, 1, "State: IDLE     ");
    OLED_ShowString(3, 1, "K1: Calibrate   ");
    OLED_ShowString(4, 1, "K2: Measure     ");

    while (1)
    {
        // 【核心】：不管板子在干嘛，第 1 行永远在实时刷新 2 秒平均功率！
        float cur_power = OLED_ShowPowerLine();
        if (cur_power > max_power) max_power = cur_power;

        key = Key_GetNum();

        if (UART1_GetRxFlag())
        {
            UART1_GetRxLine(rx_buf);
            UART1_ClearRxFlag();

            if (state == STATE_MEASURING)
            {
                float d, h, l;
                int c;
                // 完美匹配 K230 发送格式：D:xx,H:xx,L:xx,C:xx
                if (sscanf(rx_buf, "D:%f,H:%f,L:%f,C:%d", &d, &h, &l, &c) >= 3)
                {
                    // 接收到 K230 花了2秒钟收集并滤波完毕的终极准确数据！
                    last_D = d;
                    last_H = h;
                    last_L = l;
                    last_C = c;
                    
                    state = STATE_DISPLAY;
                    display_ticks = 0;
                    meas_timeout = 0;

                    // ---------------------------------------------------------
                    // 第 2 行：显示 D 和 H (保留 1 位小数)
                    // ---------------------------------------------------------
                    int d_total = (int)(last_D * 10.0f + 0.5f);
                    int d_w = d_total / 10;
                    int d_f = d_total % 10;
                    
                    int h_total = (int)(last_H * 10.0f + 0.5f);
                    int h_w = h_total / 10;
                    int h_f = h_total % 10;
                    
                    int len = sprintf(display_buf, "D:%d.%d H:%d.%d", d_w, d_f, h_w, h_f);
                    while (len < 16) display_buf[len++] = ' ';
                    display_buf[16] = '\0';
                    OLED_ShowString(2, 1, display_buf);

                    // ---------------------------------------------------------
                    // 第 3 行：显示 L (保留 1 位小数) 和 颜色识别
                    // ---------------------------------------------------------
                    const char* colors[] = {"NONE", "RED", "GREEN", "BLUE", "YELLOW", "BLACK", "WHITE", "PURPLE"};
                    int safe_c = (last_C >= 0 && last_C <= 7) ? last_C : 0;
                    
                    int l_total = (int)(last_L * 10.0f + 0.5f);
                    int l_w = l_total / 10;
                    int l_f = l_total % 10;
                    
                    len = sprintf(display_buf, "L:%d.%d %s", l_w, l_f, colors[safe_c]);
                    while (len < 16) display_buf[len++] = ' ';
                    display_buf[16] = '\0';
                    OLED_ShowString(3, 1, display_buf);

                    // ---------------------------------------------------------
                    // 第 4 行：显示 Pmax 最大功率 (保留 2 位小数)
                    // ---------------------------------------------------------
                    int pm_total = (int)(max_power * 100.0f + 0.5f);
                    int pm_w = pm_total / 100;
                    int pm_dw = pm_total % 100;
                    
                    len = sprintf(display_buf, "Pmax:%d.%02dW", pm_w, pm_dw);
                    while (len < 16) display_buf[len++] = ' ';
                    display_buf[16] = '\0';
                    OLED_ShowString(4, 1, display_buf);
                    
                    UART1_SendString("RET_IDLE\n");
                    LED2_OFF();
                }
            }
        }

        switch (state)
        {
        case STATE_IDLE:
        case STATE_WAIT_MEAS:
            if (key == 1)
            {
                // 发送 CAL_START，K230 瞬间就自动采集了 50cm 的第一张图！
                UART1_SendString("CAL_START\n");
                state = STATE_CALIBRATING;
                cal_sample_count = 1; // 完美对齐：现在已经完成了 1 个点
                
                OLED_ShowString(2, 1, "50cm Sampled!   ");
                OLED_ShowString(3, 1, "Next: 55cm      ");
                OLED_ShowString(4, 1, "Done: 1/11      ");
                LED1_ON();
            }
            else if (key == 2)
            {
                UART1_SendString("MEAS_START\n");
                state = STATE_MEASURING;
                meas_timeout = 0;
                max_power = 0.0f; // 【修复】：每次开始测量清空历史最大功率
                
                OLED_ShowString(2, 1, "Measuring...    ");
                OLED_ShowString(3, 1, "Wait 2 Seconds  ");
                OLED_ShowString(4, 1, "                ");
                LED2_ON();
            }
            break;

        case STATE_CALIBRATING:
            if (key == 1)
            {
                if (cal_sample_count < 11)
                {
                    UART1_SendString("SAMPLE\n");
                    cal_sample_count++; // 比如第二次按，变成 2

                    if (cal_sample_count <= 11)
                    {
                        // 计算刚刚采样的距离：50 + (2-1)*5 = 55cm，绝对不会跳到60！
                        uint16_t sampled_dist = 50 + (cal_sample_count - 1) * 5;
                        sprintf(display_buf, "%dcm Sampled!   ", sampled_dist);
                        OLED_ShowString(2, 1, display_buf);

                        if (cal_sample_count < 11)
                        {
                            uint16_t next_dist = sampled_dist + 5;
                            sprintf(display_buf, "Next: %dcm      ", next_dist);
                            OLED_ShowString(3, 1, display_buf);
                        }
                        else
                        {
                            OLED_ShowString(3, 1, "All points done ");
                        }

                        sprintf(display_buf, "Done: %d/11     ", cal_sample_count);
                        OLED_ShowString(4, 1, display_buf);
                    }
                }

                if (cal_sample_count >= 11)
                {
                    OLED_ShowString(2, 1, "Computing...    ");
                    OLED_ShowString(3, 1, "Saving coeffs   ");
                    OLED_ShowString(4, 1, "                ");
                    Delay_ms(500); // 留给文件保存的时间
                    
                    UART1_SendString("CAL_END\n");
                    state = STATE_WAIT_MEAS;
                    cal_sample_count = 0;
                    
                    OLED_ShowString(2, 1, "K1: Calibrate   ");
                    OLED_ShowString(3, 1, "K2: Measure     ");
                    OLED_ShowString(4, 1, "Cal complete!   ");
                    LED1_OFF();
                }
            }
            break;

        case STATE_MEASURING:
            meas_timeout++;
            // 超时时间放到 15 秒 (150*100ms)，给足 K230 拍 10 帧做滤波的时间
            if (meas_timeout > 150)
            {
                state = STATE_WAIT_MEAS;
                OLED_ShowString(2, 1, "K1: Calibrate   ");
                OLED_ShowString(3, 1, "Timeout!        ");
                OLED_ShowString(4, 1, "K2: Retry       ");
                LED2_OFF();
            }
            break;

        case STATE_DISPLAY:
            display_ticks++;
            // 结果保持显示，直到再次按下 K2 或者 15秒 后返回待机界面
            if (key == 2 || display_ticks > 150)
            {
                // 如果是手动按的 K2，直接无缝开启下一次测量
                if (key == 2) {
                    UART1_SendString("MEAS_START\n");
                    state = STATE_MEASURING;
                    meas_timeout = 0;
                    max_power = 0.0f; // 【修复】：无缝重新测量时也要清空历史最大功率
                    
                    OLED_ShowString(2, 1, "Measuring...    ");
                    OLED_ShowString(3, 1, "Wait 2 Seconds  ");
                    OLED_ShowString(4, 1, "                ");
                    LED2_ON();
                } else {
                    state = STATE_WAIT_MEAS;
                    OLED_ShowString(2, 1, "K1: Calibrate   ");
                    OLED_ShowString(3, 1, "K2: Measure     ");
                    OLED_ShowString(4, 1, "                ");
                }
            }
            break;
        }

        // 主循环 10Hz，完美配合 20帧(2秒) 环形滤波
        Delay_ms(100);
    }
}