
//  Copyright (c) 2003-2020 Xsens Technologies B.V. or subsidiaries worldwide.
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//  1.	Redistributions of source code must retain the above copyright notice,
//  	this list of conditions, and the following disclaimer.
//
//  2.	Redistributions in binary form must reproduce the above copyright notice,
//  	this list of conditions, and the following disclaimer in the documentation
//  	and/or other materials provided with the distribution.
//
//  3.	Neither the names of the copyright holders nor the names of their contributors
//  	may be used to endorse or promote products derived from this software without
//  	specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
//  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
//  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
//  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.THE LAWS OF THE NETHERLANDS
//  SHALL BE EXCLUSIVELY APPLICABLE AND ANY DISPUTES SHALL BE FINALLY SETTLED UNDER THE RULES
//  OF ARBITRATION OF THE INTERNATIONAL CHAMBER OF COMMERCE IN THE HAGUE BY ONE OR MORE
//  ARBITRATORS APPOINTED IN ACCORDANCE WITH SAID RULES.
//

#ifndef IMUPUBLISHER_H
#define IMUPUBLISHER_H

#include "packetcallback.h"
#include <tauv_msgs/XsensImuData.h>

struct ImuPublisher : public PacketCallback
{
    ros::Publisher data_pub;

    ImuPublisher(ros::NodeHandle &node)
    {
        int pub_queue_size = 10;
        data_pub = node.advertise<tauv_msgs::XsensImuData>("raw_data", pub_queue_size);
    }

    void operator()(const XsDataPacket &packet, ros::Time timestamp)
    {
        if (!packet.containsSampleTimeFine())
          return;

        const uint32_t SAMPLE_TIME_FINE_HZ = 10000UL;
        const uint32_t ONE_GHZ = 1000000000UL;
        uint32_t sec, nsec, t_fine;

        t_fine = packet.sampleTimeFine();
        sec = t_fine / SAMPLE_TIME_FINE_HZ;
        nsec = (t_fine % SAMPLE_TIME_FINE_HZ) * (ONE_GHZ / SAMPLE_TIME_FINE_HZ);

        if (packet.containsSampleTimeCoarse())
        {
            sec = packet.sampleTimeCoarse();
        }

        ros::Time sample_time(sec, nsec);

        bool quaternion_available = packet.containsOrientation();
        bool angular_velocity_available = packet.containsCalibratedGyroscopeData();
        bool linear_acceleration_available = packet.containsCalibratedAcceleration();

        geometry_msgs::Quaternion quaternion;
        if (quaternion_available)
        {
            XsQuaternion q = packet.orientationQuaternion();

            quaternion.w = q.w();
            quaternion.x = q.x();
            quaternion.y = q.y();
            quaternion.z = -q.z();
        }

        geometry_msgs::Vector3 angular_velocity;
        if (angular_velocity_available)
        {
            XsVector a = packet.calibratedGyroscopeData();
            angular_velocity.x = a[0];
            angular_velocity.y = -a[1];
            angular_velocity.z = -a[2];
        }

        geometry_msgs::Vector3 linear_acceleration;
        if (linear_acceleration_available)
        {
            XsVector a = packet.calibratedAcceleration();
            linear_acceleration.x = a[0];
            linear_acceleration.y = -a[1];
            linear_acceleration.z = -a[2];
        }

        uint32_t status = packet.status();
        bool triggered_dvl = (status >> 22) & 1;

        if (quaternion_available && angular_velocity_available && linear_acceleration_available)
        {
            tauv_msgs::XsensImuData data_msg;

            data_msg.header.stamp = timestamp;

            data_msg.ros_time = timestamp;
            data_msg.imu_time = sample_time;

            data_msg.triggered_dvl = triggered_dvl;

            data_msg.orientation = quaternion;
            data_msg.angular_velocity = angular_velocity;
            data_msg.linear_acceleration = linear_acceleration;

            data_pub.publish(data_msg);
        }
    }
};

#endif
