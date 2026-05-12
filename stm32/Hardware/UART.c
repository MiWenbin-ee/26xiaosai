#include "UART.h"

#define UART1_RX_BUF_SIZE 64

static char uart1_rx_buf[UART1_RX_BUF_SIZE];
static uint8_t uart1_rx_index = 0;
static uint8_t uart1_rx_complete = 0;

void UART1_Init(uint32_t baud)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1 | RCC_APB2Periph_GPIOA, ENABLE);

	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_Init(GPIOA, &GPIO_InitStructure);

	USART_InitTypeDef USART_InitStructure;
	USART_InitStructure.USART_BaudRate = baud;
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	USART_Init(USART1, &USART_InitStructure);

	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);

	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
	NVIC_Init(&NVIC_InitStructure);

	USART_Cmd(USART1, ENABLE);
}

void UART1_SendByte(uint8_t data)
{
	while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
	USART_SendData(USART1, data);
	while (USART_GetFlagStatus(USART1, USART_FLAG_TC) == RESET);
}

void UART1_SendString(char *str)
{
	while (*str)
	{
		UART1_SendByte(*str++);
	}
}

uint8_t UART1_GetRxFlag(void)
{
	return uart1_rx_complete;
}

void UART1_ClearRxFlag(void)
{
	uart1_rx_complete = 0;
	uart1_rx_index = 0;
}

void UART1_GetRxLine(char *buf)
{
	uint8_t i;
	for (i = 0; i < uart1_rx_index; i++)
	{
		buf[i] = uart1_rx_buf[i];
	}
	buf[i] = '\0';
}

void USART1_IRQHandler(void)
{
	if (USART_GetITStatus(USART1, USART_IT_RXNE) == SET)
	{
		uint8_t data = USART_ReceiveData(USART1);

		if (data == '\n')
		{
			uart1_rx_buf[uart1_rx_index] = '\0';
			uart1_rx_complete = 1;
		}
		else if (data != '\r' && uart1_rx_index < UART1_RX_BUF_SIZE - 1)
		{
			uart1_rx_buf[uart1_rx_index++] = data;
		}
	}
	USART_ClearITPendingBit(USART1, USART_IT_RXNE);
}
