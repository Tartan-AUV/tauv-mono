################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Modules/ESCModule.cpp \
../Modules/Eth100HzModule.cpp \
../Modules/Eth50HzModule.cpp \
../Modules/MS5837Module.cpp \
../Modules/MTI300Module.cpp 

OBJS += \
./Modules/ESCModule.o \
./Modules/Eth100HzModule.o \
./Modules/Eth50HzModule.o \
./Modules/MS5837Module.o \
./Modules/MTI300Module.o 

CPP_DEPS += \
./Modules/ESCModule.d \
./Modules/Eth100HzModule.d \
./Modules/Eth50HzModule.d \
./Modules/MS5837Module.d \
./Modules/MTI300Module.d 


# Each subdirectory must supply rules for building sources it contributes
Modules/%.o Modules/%.su Modules/%.cyclo: ../Modules/%.cpp Modules/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m7 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F767xx -c -I../Core/Inc -I../LWIP/App -I../LWIP/Target -I../Middlewares/Third_Party/LwIP/src/include -I../Middlewares/Third_Party/LwIP/system -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Middlewares/Third_Party/FreeRTOS/Source/include -I../Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS -I../Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1 -I../Drivers/BSP/Components/lan8742 -I../Middlewares/Third_Party/LwIP/src/include/netif/ppp -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Middlewares/Third_Party/LwIP/src/include/lwip -I../Middlewares/Third_Party/LwIP/src/include/lwip/apps -I../Middlewares/Third_Party/LwIP/src/include/lwip/priv -I../Middlewares/Third_Party/LwIP/src/include/lwip/prot -I../Middlewares/Third_Party/LwIP/src/include/netif -I../Middlewares/Third_Party/LwIP/src/include/compat/posix -I../Middlewares/Third_Party/LwIP/src/include/compat/posix/arpa -I../Middlewares/Third_Party/LwIP/src/include/compat/posix/net -I../Middlewares/Third_Party/LwIP/src/include/compat/posix/sys -I../Middlewares/Third_Party/LwIP/src/include/compat/stdc -I../Middlewares/Third_Party/LwIP/system/arch -I../Drivers/CMSIS/Include -I../Modules -I../Util -I../Tasks -I../Drivers/VESC -I../Drivers/XSens -I../Drivers/MS5837 -I../FlatBuffers -I/opt/homebrew/include -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -std=c++17 -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Modules

clean-Modules:
	-$(RM) ./Modules/ESCModule.cyclo ./Modules/ESCModule.d ./Modules/ESCModule.o ./Modules/ESCModule.su ./Modules/Eth100HzModule.cyclo ./Modules/Eth100HzModule.d ./Modules/Eth100HzModule.o ./Modules/Eth100HzModule.su ./Modules/Eth50HzModule.cyclo ./Modules/Eth50HzModule.d ./Modules/Eth50HzModule.o ./Modules/Eth50HzModule.su ./Modules/MS5837Module.cyclo ./Modules/MS5837Module.d ./Modules/MS5837Module.o ./Modules/MS5837Module.su ./Modules/MTI300Module.cyclo ./Modules/MTI300Module.d ./Modules/MTI300Module.o ./Modules/MTI300Module.su

.PHONY: clean-Modules

