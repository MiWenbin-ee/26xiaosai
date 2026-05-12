#ifndef __UART_H
#define __UART_H
#include "stm32f10x.h"

void UART1_Init(uint32_t baud);
void UART1_SendString(char *str);
void UART1_SendByte(uint8_t data);
uint8_t UART1_GetRxFlag(void);
void UART1_ClearRxFlag(void);
void UART1_GetRxLine(char *buf);

#endif
