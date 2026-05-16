#ifndef __ADC_H__
#define __ADC_H__
#include "stm32f10x.h"
// 采样窗口大小 (每个通道连续采样的次数)
// 值越大，滤波效果越好，但更新会有微小延迟
#define ADC_SAMPLE_NUM 100 

/* 函数声明 ------------------------------------------------------------------*/
void ADCDMA_Init(void);
float Get_Current_Amps(void);
#endif