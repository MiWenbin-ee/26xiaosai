#include "stm32f10x.h"
#include "Delay.h"
#include "OLED.h"
#include "ADC.h"
#include "Key.h"
#include "LED.h"
#include "UART.h"
#include <stdio.h>

enum {
	STATE_IDLE,
	STATE_CALIBRATING,
	STATE_WAIT_MEAS,
	STATE_MEASURING,
	STATE_DISPLAY
};

int main(void)
{
	uint8_t state = STATE_IDLE;
	uint8_t key;
	char rx_buf[32];
	char display_buf[24];
	float last_D = 0.0f, last_H = 0.0f;
	uint16_t display_ticks = 0;
	uint16_t meas_timeout = 0;

	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);

	OLED_Init();
	LED_Init();
	Key_Init();
	ADCDMA_Init();
	ADC_Pot_Init();
	UART1_Init(115200);

	LED1_OFF();
	LED2_OFF();

	OLED_ShowString(1, 1, "State: IDLE     ");
	OLED_ShowString(2, 1, "KEY1: Calibrate ");
	OLED_ShowString(3, 1, "                ");
	OLED_ShowString(4, 1, "                ");

	while (1)
	{
		key = Key_GetNum();

		if (UART1_GetRxFlag())
		{
			UART1_GetRxLine(rx_buf);
			UART1_ClearRxFlag();

			if (state == STATE_MEASURING)
			{
				float d, h;
				if (sscanf(rx_buf, "D:%f,H:%f", &d, &h) == 2)
				{
					last_D = d;
					last_H = h;
					state = STATE_DISPLAY;
					display_ticks = 0;
					meas_timeout = 0;

					OLED_ShowString(1, 1, "State: RESULT   ");
					sprintf(display_buf, "D: %.1f cm     ", last_D);
					OLED_ShowString(2, 1, display_buf);
					sprintf(display_buf, "H: %.1f cm     ", last_H);
					OLED_ShowString(3, 1, display_buf);
					OLED_ShowString(4, 1, "KEY2: Next      ");

					UART1_SendString("RET_IDLE\n");
					LED2_OFF();
				}
			}
		}

		switch (state)
		{
		case STATE_IDLE:
			if (key == 1)
			{
				UART1_SendString("CAL_START\n");
				state = STATE_CALIBRATING;
				OLED_ShowString(1, 1, "State: CALIBRAT ");
				OLED_ShowString(2, 1, "Adjust pot for K ");
				OLED_ShowString(3, 1, "KEY1: End Cal   ");
				OLED_ShowString(4, 1, "                ");
				LED1_ON();
			}
			break;

		case STATE_CALIBRATING:
			{
				float k_H = Get_Potentiometer_K();
				sprintf(display_buf, "K:%.6f       ", k_H);
				OLED_ShowString(4, 1, display_buf);
				sprintf(display_buf, "K:%.6f\n", k_H);
				UART1_SendString(display_buf);
			}
			if (key == 1)
			{
				UART1_SendString("CAL_END\n");
				state = STATE_WAIT_MEAS;
				OLED_ShowString(1, 1, "State: WAITING  ");
				OLED_ShowString(2, 1, "KEY2: Measure   ");
				OLED_ShowString(3, 1, "                ");
				OLED_ShowString(4, 1, "                ");
				LED1_OFF();
			}
			break;

		case STATE_WAIT_MEAS:
			if (key == 2)
			{
				UART1_SendString("MEAS_START\n");
				state = STATE_MEASURING;
				meas_timeout = 0;
				OLED_ShowString(1, 1, "State: MEASURNG ");
				OLED_ShowString(2, 1, "Measuring...    ");
				OLED_ShowString(3, 1, "                ");
				OLED_ShowString(4, 1, "                ");
				LED2_ON();
			}
			break;

		case STATE_MEASURING:
			meas_timeout++;
			if (meas_timeout > 100)
			{
				state = STATE_WAIT_MEAS;
				OLED_ShowString(1, 1, "State: WAITING  ");
				OLED_ShowString(2, 1, "Timeout!        ");
				OLED_ShowString(3, 1, "                ");
				OLED_ShowString(4, 1, "                ");
				LED2_OFF();
			}
			break;

		case STATE_DISPLAY:
			display_ticks++;
			if (key == 2 || display_ticks > 100)
			{
				state = STATE_WAIT_MEAS;
				OLED_ShowString(1, 1, "State: WAITING  ");
				OLED_ShowString(2, 1, "KEY2: Measure   ");
				OLED_ShowString(3, 1, "                ");
				OLED_ShowString(4, 1, "                ");
			}
			break;
		}

		Delay_ms(100);
	}
}
