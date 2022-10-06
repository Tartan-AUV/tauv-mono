set(CMAKE_CXX_STANDARD 17)
library(controller controller.cpp)
library(dynamics dynamics.cpp)
node(controller_cpp main.cpp controller dynamics tauv_alarms tauv_utils)