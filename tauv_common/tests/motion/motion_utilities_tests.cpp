// Motion Utilites Tests 
// 

#include <motion/motion_utilities.h>

#include <geometry_msgs/Twist.h>
#include "ros/ros.h"
#include <gtest/gtest.h>
#include "units/units.h"

static constexpr double EPSILON = 1e-6;

TEST(MotionUtilitesTests, ForwardTest)
{
  ros::init(0, NULL, "motion_utilities_test");
  ros::NodeHandle nh("~");
  Mover mover(nh);
  EXPECT_EQ(3, 3);
}

int main(int argc, char** argv) {
    return RUN_ALL_TESTS();
}
