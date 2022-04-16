add_custom_target(
    generate_alarms ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/alarms.py 
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/alarms.py
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_alarms.py ${CMAKE_CURRENT_SOURCE_DIR}/alarms.yaml -o ${CMAKE_CURRENT_SOURCE_DIR}/alarms.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/alarms.yaml ${CMAKE_CURRENT_SOURCE_DIR}/generate_alarms.py
    VERBATIM
)