execute_process(COMMAND "/home/tom/workspaces/tauv_ws/build/tauv_vehicle/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/tom/workspaces/tauv_ws/build/tauv_vehicle/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
