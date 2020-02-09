# Macro for declaring cpp unit tests
# First argument is the executable name
macro(cpp_test name file)
    catkin_add_gtest(${name} ${file})
    target_link_libraries(${name} gtest gmock ${catkin_LIBRARIES})
endmacro(cpp_test)

macro(ros_node_cpp_test name ros_file cpp_file)
  add_rostest_gtest(name
                    ros_file
                    cpp_file)
  target_link_libraries(name ${catkin_LIBRARIES})
endmacro(ros_node_cpp_test)
