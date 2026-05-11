#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "OLED.h"
#include "ADC.h"
#include <stdio.h> 
char display_buf[20];
float current_is = 0.0;
float power = 0.0;
int main(void)
{
	OLED_Init();
	ADCDMA_Init();
	OLED_ShowString(1, 1, "Sys Status: OK");

    while(1)
    {
        // 1. 获取滤波后的真实电流
        current_is = Get_Current_Amps();
        
        // 2. 计算功耗 (题目规定 U = 5V)
        power = 5.0f * current_is;
        
        // 3. 格式化并显示到 OLED (假设使用 sprintf 组合字符串)
        sprintf(display_buf, "I: %.3f A", current_is);
        OLED_ShowString(2, 1, display_buf);
        
        sprintf(display_buf, "P: %.2f W", power);
        OLED_ShowString(3, 1, display_buf);
        
        Delay_ms(200); // 刷新率不要太高，人眼看不清
    }
}
