#include "stm32f10x.h"

#define ADC_SAMPLE_NUM 100 // 采样窗口大小

// 定义一个数组用于 DMA 自动搬运 ADC 数据
uint16_t ADC_Value[ADC_SAMPLE_NUM]; 

void ADCDMA_Init(void)
{
    // 1. 开启时钟
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_ADC1 | RCC_APB2Periph_GPIOA, ENABLE);
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1, ENABLE);
    RCC_ADCCLKConfig(RCC_PCLK2_Div6); // 设置 ADC 分频，最大不超过 14MHz

    // 2. 配置 GPIO (PA0 模拟输入)
    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // 3. 配置 ADC1
    ADC_InitTypeDef ADC_InitStructure;
    ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;      // 独立模式
    ADC_InitStructure.ADC_ScanConvMode = DISABLE;           // 单通道不扫描
    ADC_InitStructure.ADC_ContinuousConvMode = ENABLE;      // 连续转换模式开启
    ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None; // 软件触发
    ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;  // 右对齐
    ADC_InitStructure.ADC_NbrOfChannel = 1;                 // 通道数 1
    ADC_Init(ADC1, &ADC_InitStructure);
    
    // 设置规则组通道，采样时间尽量拉长以保证精度
    ADC_RegularChannelConfig(ADC1, ADC_Channel_0, 1, ADC_SampleTime_239Cycles5);

    // 4. 配置 DMA1 通道1
    DMA_InitTypeDef DMA_InitStructure;
    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)&ADC1->DR; // 外设地址：ADC数据寄存器
    DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)ADC_Value;     // 内存地址：我们定义的数组
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralSRC;              // 方向：外设到内存
    DMA_InitStructure.DMA_BufferSize = ADC_SAMPLE_NUM;              // 传输大小
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;// 外设地址不增
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;         // 内存地址自增
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_HalfWord; // 16位
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_HalfWord;         // 16位
    DMA_InitStructure.DMA_Mode = DMA_Mode_Circular;                 // 循环模式 (关键)
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;
    DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(DMA1_Channel1, &DMA_InitStructure);

    // 5. 使能各个模块
    DMA_Cmd(DMA1_Channel1, ENABLE);
    ADC_DMACmd(ADC1, ENABLE);
    ADC_Cmd(ADC1, ENABLE);

    // 6. ADC 校准 (必须执行)
    ADC_ResetCalibration(ADC1);
    while (ADC_GetResetCalibrationStatus(ADC1) == SET);
    ADC_StartCalibration(ADC1);
    while (ADC_GetCalibrationStatus(ADC1) == SET);

    // 7. 软件触发 ADC 开始转换
    ADC_SoftwareStartConvCmd(ADC1, ENABLE);
}

// 获取滤波后的 ADC 平均值
uint16_t Get_ADC_Average(void)
{
    uint32_t sum = 0;
    
    // 简单地对 DMA 缓冲区内的数据求和
    for(int i = 0; i < ADC_SAMPLE_NUM; i++)
    {
        sum += ADC_Value[i];
    }
    
    return (uint16_t)(sum / ADC_SAMPLE_NUM);
}

// 进一步：将 ADC 原始值转换为真实电流值 (A)
// 假设你使用的采样电阻 Rs = 0.1 欧姆，INA270 放大倍数为 14
float Get_Current_Amps(void)
{
    uint16_t adc_raw = Get_ADC_Average();
    
    // 1. 将 ADC 值转换为电压值 (STM32 基准电压通常为 3.3V，12位 ADC)
    float vout = ((float)adc_raw / 4095.0f) * 3.3f;
    
    // 2. 根据 INA270 公式反推电流: Vout = I * Rs * 14
    // 所以 I = Vout / (Rs * 14)
    float current = vout / (0.1f * 14.0f); 
    
    return current;
}

void ADC_Pot_Init(void)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
}

float Get_Potentiometer_K(void)
{
	uint16_t adc_val = 0;
	uint8_t i;

	for (i = 0; i < 10; i++)
	{
		ADC_RegularChannelConfig(ADC1, ADC_Channel_3, 1, ADC_SampleTime_239Cycles5);
		ADC_SoftwareStartConvCmd(ADC1, ENABLE);
		while (ADC_GetFlagStatus(ADC1, ADC_FLAG_EOC) == RESET);
		adc_val += ADC_GetConversionValue(ADC1);
	}
	adc_val /= 10;

	ADC_RegularChannelConfig(ADC1, ADC_Channel_0, 1, ADC_SampleTime_239Cycles5);

	return 0.0001f + ((float)adc_val / 4095.0f) * 0.0049f;
}