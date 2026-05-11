#ifndef __LightSensor_H
#define __LightSensor_H
#include "stm32f10x.h"
void LightSensor_Init(void);
uint8_t LightSensor_Get(void);
#endif
