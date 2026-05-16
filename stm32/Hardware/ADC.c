#include "ADC.h"
#include "stm32f10x.h"
/* 私有变量 ------------------------------------------------------------------*/
__IO uint16_t ADC_Value[ADC_SAMPLE_NUM];

/* 私有函数声明 --------------------------------------------------------------*/
static uint16_t Get_Current_ADC_Average(void);


/**
  * @brief  初始化 ADC1 和 DMA，单通道连续转换 + DMA 循环覆盖
  * @param  无
  * @retval 无
  */
void ADCDMA_Init(void)
{
    // 1. 开启时钟：ADC1、GPIOA、DMA1
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1 | RCC_APB2Periph_GPIOA, ENABLE);
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
    RCC_ADCCLKConfig(RCC_PCLK2_Div6);

    // 2. 配置 GPIO (PA0 测电流)
    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // 3. 配置 ADC1 (单通道连续模式)
    ADC_InitTypeDef ADC_InitStructure;
    ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
    ADC_InitStructure.ADC_ScanConvMode = DISABLE;
    ADC_InitStructure.ADC_ContinuousConvMode = ENABLE;
    ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
    ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
    ADC_InitStructure.ADC_NbrOfChannel = 1;
    ADC_Init(ADC1, &ADC_InitStructure);

    ADC_RegularChannelConfig(ADC1, ADC_Channel_0, 1, ADC_SampleTime_239Cycles5);

    // 4. 先使能 ADC 并校准 (必须先上电才能校准)
    ADC_Cmd(ADC1, ENABLE);
    ADC_ResetCalibration(ADC1);
    while (ADC_GetResetCalibrationStatus(ADC1) == SET);
    ADC_StartCalibration(ADC1);
    while (ADC_GetCalibrationStatus(ADC1) == SET);

    // 5. 配置 DMA1 通道1
    DMA_InitTypeDef DMA_InitStructure;
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&ADC1->DR;
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)ADC_Value;
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;
    DMA_InitStructure.DMA_BufferSize = ADC_SAMPLE_NUM;
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord;
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;
    DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(DMA1_Channel1, &DMA_InitStructure);

    // 6. 使能 DMA (先开 ADC_DMA，再开 DMA 通道)
    ADC_DMACmd(ADC1, ENABLE);
    DMA_Cmd(DMA1_Channel1, ENABLE);

    // 7. 软件触发首轮转换 (后续硬件自动循环)
    ADC_SoftwareStartConvCmd(ADC1, ENABLE);
}

/**
  * @brief  获取滤波后的电流 ADC 平均值 (内部调用)
  * @retval 12位 ADC 均值 (0~4095)
  */
static uint16_t Get_Current_ADC_Average(void)
{
    uint32_t sum = 0;
    
    // 对 DMA 缓冲区内的通道0数据求和
    for(int i = 0; i < ADC_SAMPLE_NUM; i++)
    {
        sum += ADC_Value[i];
    }
    
    return (uint16_t)(sum / ADC_SAMPLE_NUM);
}

/**
  * @brief  获取真实的电流值 (安培 A)
  * @retval 电流值 (float)
  */
float Get_Current_Amps(void)
{
    uint16_t adc_raw = Get_Current_ADC_Average();
    
    // 1. 将 ADC 值转换为电压值 (STM32 基准电压通常为 3.3V，12位 ADC)
    float vout = ((float)adc_raw / 4095.0f) * 3.3f; 
    
    float current = vout / (0.1f * 15.9f);
    
    return current;
}
