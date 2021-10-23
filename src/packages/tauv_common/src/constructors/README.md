# ROS Message Constructors

An annoying feature of ROS messages is that the only "official" way to construct them is with a default empty constructor, followed by filling in the fields of the message type. Given that this is verbose and annoying for larger messages, we maintain a set of convenience constructors to help out.

This turns
```
geometry_msgs::PoseStamped msg;
msg.header.seq = 0;
msg.header.time = ros::Time::now();
msg.header.frame_id = "";
msg.pose.position.x = 0;
msg.pose.position.y = 0;
msg.pose.position.z = 0;
msg.pose.orientation.x = 0;
msg.pose.orientation.y = 0;
msg.pose.orientation.z = 0;
msg.pose.orientation.w = 0;
```
into
```
geometry_msgs::PoseStamped msg = PoseStamped(Header(0, ros::Time::now(), ""), Point(0,0,0), Quaternion(0,0,0,0));
```

For convenience, a default constructor is defined that "zeros" all fields of the message. So this operation can be written as
``
geometry_msgs::PoseStamped msg = PoseStamped();




## Guidelines
* A zero constructor should be provided, for example `Vector3()` should create a message with all fields "zeroed" (or whatever the equivalent is for the type).
* Top level message members should be the only parameters to the constructors. For example, `PoseStamped(std_msgs::Header, geometry_msgs::Pose)` is acceptable, but `PoseStamped(std_msgs::Header, geometry_msgs::Point, geometry_msgs::Quaternion)` is not.
* You should create constructors for every non primitive sub type in a message. So `geometry_msgs::PoseStamped` requires `geometry_msgs::Pose`,`geometry_msgs::Point`, and `geometry_msgs::Quaternion` all be defined.
