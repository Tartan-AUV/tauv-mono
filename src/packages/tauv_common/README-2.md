# Package `tauv_common`
This package contains the modules common to all TAUV vehicles. That is, this package is submarine-agnostic and includes general-purpose functionality like state estimation, health monitoring, motion planners, and sensor driver bases.

## Submodule `tauv_msgs`
## Submodule `tauv_util`
## Submodule `controllers`
## Submodule `sensors`

Sensors are abstracted away from the specific vehicle using `Communication` classes and `Sensor` classes. `Communication` classes define general-purpose classes for different communication protocols like serial or ethernet. `Sensor` classes are built on top of `Communication` classes and inherent from them, later implementing sensor-specific methods.

`Sensor` (and thus `Communication`) instances require that a `ros::NodeHandler` be passed. This is so that all sensor and communication classes can be synchronized throughout time, and their lifecycle is defined by the state of the `ros::NodeHandler`.

### Serial Devices
This class represents a general serial device. 

```c++
#include<thread>

using namespace std;

class SerialDevice {
	protected:
		string name; // device name
		string port; // absolute file path to serial port in use
		int32_t baudRate; // device baud rate
		
		size_t msgLenBytes; // length of serial messages in bytes
		size_t msgBufferLength; // length of message buffer
		char[msgLenBytes] latestMsg; // stores the latest message
		char[msgLenBytes][msgBufferLength] latestMsgBuffer; // stores a buffer of latest messages
		mutex latestMsgLock; // mutex lock on accessing the latest message

		ros::NodeHandle nh;
		
	private:
		void readSerialThreaded() {
			while(nh.ok()) {
				latestMsgLock.lock(); // acquire the mutex
				latestMsg = readline();
				latestMsgLock.unlock(); // release the mutex
			}
		}
		
	public:
		SerialDevice(_name, _port, _baudRate, _msgLenBytes, _msgBufferLength) {
			name = _name;
			port = _port;
			baudRate = _baudRate;
			msgLenBytes = _msgLenBytes;
			msgBufferLength = _msgBufferLength;
			
			thread thread_object(readSerialThreaded);
		}
		
		char[] getLatestMsg() {
			char[] _latestMsg;
			
			latestMsgLock.lock(); // acquire the mutex
			_latestMsg = _latestMsg;
			latestMsgLock.unlock(); // release the mutex
			
			return _latestMsg;
		}
}
```
### IMU
```c++
class IMU: public SerialDevice {
	private:
		
	public:
		void publishQuaternion();
}
```
### Depth Sensor
### DVL
## Submodule `hw_devices`
### Thruster
### Servo
### Battery