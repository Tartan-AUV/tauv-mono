################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Drivers/VESC/VESC_UART.cpp \
../Drivers/VESC/buffer.cpp \
../Drivers/VESC/crc.cpp 

OBJS += \
./Drivers/VESC/VESC_UART.o \
./Drivers/VESC/buffer.o \
./Drivers/VESC/crc.o 

CPP_DEPS += \
./Drivers/VESC/VESC_UART.d \
./Drivers/VESC/buffer.d \
./Drivers/VESC/crc.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/VESC/%.o Drivers/VESC/%.su Drivers/VESC/%.cyclo: ../Drivers/VESC/%.cpp Drivers/VESC/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m7 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F767xx -c -I../Core/Inc -I../LWIP/App -I../LWIP/Target -I../Middlewares/Third_Party/LwIP/src/include -I../Middlewares/Third_Party/LwIP/system -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Middlewares/Third_Party/FreeRTOS/Source/include -I../Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS -I../Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1 -I../Drivers/BSP/Components/lan8742 -I../Middlewares/Third_Party/LwIP/src/include/netif/ppp -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Middlewares/Third_Party/LwIP/src/include/lwip -I../Middlewares/Third_Party/LwIP/src/include/lwip/apps -I../Middlewares/Third_Party/LwIP/src/include/lwip/priv -I../Middlewares/Third_Party/LwIP/src/include/lwip/prot -I../Middlewares/Third_Party/LwIP/src/include/netif -I../Middlewares/Third_Party/LwIP/src/include/compat/posix -I../Middlewares/Third_Party/LwIP/src/include/compat/posix/arpa -I../Middlewares/Third_Party/LwIP/src/include/compat/posix/net -I../Middlewares/Third_Party/LwIP/src/include/compat/posix/sys -I../Middlewares/Third_Party/LwIP/src/include/compat/stdc -I../Middlewares/Third_Party/LwIP/system/arch -I../Drivers/CMSIS/Include -I../Modules -I../Util -I../Tasks -I../Drivers/VESC -I../Drivers/XSens -I../Drivers/MS5837 -I../FlatBuffers -I/opt/homebrew/include -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -std=c++17 -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-VESC

clean-Drivers-2f-VESC:
	-$(RM) ./Drivers/VESC/VESC_UART.cyclo ./Drivers/VESC/VESC_UART.d ./Drivers/VESC/VESC_UART.o ./Drivers/VESC/VESC_UART.su ./Drivers/VESC/buffer.cyclo ./Drivers/VESC/buffer.d ./Drivers/VESC/buffer.o ./Drivers/VESC/buffer.su ./Drivers/VESC/crc.cyclo ./Drivers/VESC/crc.d ./Drivers/VESC/crc.o ./Drivers/VESC/crc.su

.PHONY: clean-Drivers-2f-VESC

