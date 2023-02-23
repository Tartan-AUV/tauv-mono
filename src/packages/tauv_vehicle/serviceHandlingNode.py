# dependencies must handle ROS, serial
import rospy
import serial
from request_response_comm_protocol.srv import sensorName, currValue


# request type should be a string and response can just be an int

# check dependencies in github packages.xml, if it exists

# define a single service for now (let's just say turn a light on on a circuit)

# shall implement a simple ROS node which turns a light on on Arduino, 
# "upload" it to the repo and test it on the Arduino

# initialization of the service

class sensorValue_server:
    def __init__(self):
        rospy.init_node('sensor_value', anonymous=False)
        self.srv = rospy.Service('give_sensor_value', currValue, self.sensorCallback)
        self.response = getSensorValue() # the pyserial bit
    
    def sensorCallback(self, request):
        resp = replyResponse()
        resp.sensorResponse = self.response

if __name__ == '__main__':
    s = sensorValue_server()
    rospy.spin()
        

def turn_light_on(req):
    return some kind of service class"""(req.a + req.b) # figure out different way of unpacking args

def turn_light_on_server():
    rospy.init_node('turn_light_on_server')
    s = rospy.Service('turn_light_on', """some kind of service class""", turn_light_on)
    rospy.spin()
    

################### represents sample ROS code for learning how to build ROS service #########



#! /usr/bin/env python
import rospy                                      # the main module for ROS-python programs
#from std_srvs.srv import Trigger, TriggerResponse # we are creating a 'Trigger service'...
                                                  # ...Other types are available, and you can create
                                                  # custom types

# here Trigger is the request type and TriggerResponse is the response type
# which I must suit to my purposes (but then must figure out how to define?)

# for now request type is Light and response type is LightResponse






def light_response(request):
    ''' 
    Callback function used by the service server to process
    requests from clients. It returns a TriggerResponse
    '''
    ser = serial.Serial('COM4', 9800, timeout=1) # some kind of serial initialization
    line = ser.readline() # ideally this should be the light state
    if line:
        string = line.decode()  # convert the byte string to a unicode string
        num = int(string) # 0 for off 1 for on ideally
    ser.close()
    return LightResponse(
        success=True,
        message=string(num)
    )

rospy.init_node('sos_service')                     # initialize a ROS node
my_service = rospy.Service(                        # create a service, specifying its name,
    '/fake_911', Light, light_response         # type, and callback
)
rospy.spin()                                       # Keep the program from exiting, until Ctrl + C is pressed





# put this on the repo
# figure out request/response type creation
# figure out dependency location
# figure out serial initialization
# figure out Arduino side ()


