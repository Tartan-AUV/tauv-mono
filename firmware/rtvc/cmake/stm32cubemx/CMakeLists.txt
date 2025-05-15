cmake_minimum_required(VERSION 3.22)
# Enable CMake support for ASM and C languages
enable_language(C ASM)
# STM32CubeMX generated symbols (macros)
set(MX_Defines_Syms 
	USE_HAL_DRIVER 
	STM32F767xx
    $<$<CONFIG:Debug>:DEBUG>
)

# STM32CubeMX generated include paths
set(MX_Include_Dirs
    ${CMAKE_SOURCE_DIR}/Core/Inc
    ${CMAKE_SOURCE_DIR}/LWIP/App
    ${CMAKE_SOURCE_DIR}/LWIP/Target
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/system
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Inc
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Inc/Legacy
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/include
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1
    ${CMAKE_SOURCE_DIR}/Drivers/BSP/Components/lan8742
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/netif/ppp
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Device/ST/STM32F7xx/Include
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/lwip
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/lwip/apps
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/lwip/priv
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/lwip/prot
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/netif
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/compat/posix
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/compat/posix/arpa
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/compat/posix/net
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/compat/posix/sys
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/include/compat/stdc
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/system/arch
    ${CMAKE_SOURCE_DIR}/Drivers/CMSIS/Include
)

# STM32CubeMX generated application sources
set(MX_Application_Src
    ${CMAKE_SOURCE_DIR}/LWIP/Target/ethernetif.c
    ${CMAKE_SOURCE_DIR}/LWIP/App/lwip.c
    ${CMAKE_SOURCE_DIR}/Core/Src/main.cpp
    ${CMAKE_SOURCE_DIR}/Core/Src/freertos.c
    ${CMAKE_SOURCE_DIR}/Core/Src/stm32f7xx_it.c
    ${CMAKE_SOURCE_DIR}/Core/Src/stm32f7xx_hal_msp.c
    ${CMAKE_SOURCE_DIR}/Core/Src/stm32f7xx_hal_timebase_tim.c
    ${CMAKE_SOURCE_DIR}/Core/Src/sysmem.c
    ${CMAKE_SOURCE_DIR}/Core/Src/syscalls.c
    ${CMAKE_SOURCE_DIR}/startup_stm32f767xx.s
)

# STM32 HAL/LL Drivers
set(STM32_Drivers_Src
    ${CMAKE_SOURCE_DIR}/Core/Src/system_stm32f7xx.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_tim.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_tim_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_rcc_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_flash.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_flash_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_gpio.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_dma_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_pwr_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_cortex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_i2c.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_i2c_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_exti.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_eth.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_can.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_uart.c
    ${CMAKE_SOURCE_DIR}/Drivers/STM32F7xx_HAL_Driver/Src/stm32f7xx_hal_uart_ex.c
    ${CMAKE_SOURCE_DIR}/Drivers/BSP/Components/lan8742/lan8742.c
)

# Drivers Midllewares


set(FreeRTOS_Src
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/croutine.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/event_groups.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/list.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/queue.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/stream_buffer.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/tasks.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/timers.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS/cmsis_os.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/portable/MemMang/heap_4.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1/port.c
)
set(LwIP_Src
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/auth.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/ccp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/chap_ms.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/chap-md5.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/chap-new.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/demand.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/eap.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/eui64.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/fsm.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/ipcp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/ipv6cp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/lcp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/magic.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/mppe.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/multilink.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/ppp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/pppapi.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/pppcrypt.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/pppoe.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/pppol2tp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/pppos.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/upap.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/utils.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/vj.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/bridgeif.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/bridgeif_fdb.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ethernet.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/lowpan6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/lowpan6_ble.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/lowpan6_common.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/slipif.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/zepif.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/netif/ppp/ecp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/api_lib.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/api_msg.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/err.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/if_api.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/netbuf.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/netdb.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/netifapi.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/sockets.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/api/tcpip.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/altcp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/altcp_alloc.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/altcp_tcp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/def.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/dns.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/inet_chksum.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/init.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ip.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/mem.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/memp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/netif.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/pbuf.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/raw.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/stats.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/sys.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/tcp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/tcp_in.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/tcp_out.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/timeouts.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/udp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/autoip.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/dhcp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/etharp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/icmp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/igmp.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/ip4.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/ip4_addr.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv4/ip4_frag.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/dhcp6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/ethip6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/icmp6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/inet6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/ip6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/ip6_addr.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/ip6_frag.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/mld6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/core/ipv6/nd6.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/system/OS/sys_arch.c
    ${CMAKE_SOURCE_DIR}/Middlewares/Third_Party/LwIP/src/apps/mqtt/mqtt.c
)

# Link directories setup
set(MX_LINK_DIRS

)
# Project static libraries
set(MX_LINK_LIBS 
    STM32_Drivers
    FreeRTOS
	LwIP
	
)
# Interface library for includes and symbols
add_library(stm32cubemx INTERFACE)
target_include_directories(stm32cubemx INTERFACE ${MX_Include_Dirs})
target_compile_definitions(stm32cubemx INTERFACE ${MX_Defines_Syms})

# Create STM32_Drivers static library
add_library(STM32_Drivers OBJECT)
target_sources(STM32_Drivers PRIVATE ${STM32_Drivers_Src})
target_link_libraries(STM32_Drivers PUBLIC stm32cubemx)


# Create FreeRTOS static library
add_library(FreeRTOS OBJECT)
target_sources(FreeRTOS PRIVATE ${FreeRTOS_Src})
target_link_libraries(FreeRTOS PUBLIC stm32cubemx)

# Create LwIP static library
add_library(LwIP OBJECT)
target_sources(LwIP PRIVATE ${LwIP_Src})
target_link_libraries(LwIP PUBLIC stm32cubemx)

# Add STM32CubeMX generated application sources to the project
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${MX_Application_Src})

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE ${MX_LINK_DIRS})

# Add libraries to the project
target_link_libraries(${CMAKE_PROJECT_NAME} ${MX_LINK_LIBS})

# Add the map file to the list of files to be removed with 'clean' target
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_PROJECT_NAME}.map)

# Validate that STM32CubeMX code is compatible with C standard
if((CMAKE_C_STANDARD EQUAL 90) OR (CMAKE_C_STANDARD EQUAL 99))
    message(ERROR "Generated code requires C11 or higher")
endif()
