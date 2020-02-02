# Controllers
This folder is where all controllers live. This readme serves as a basic tutorial for adding a controller.

# List of controllers
## Keyboard_Controller
This is a basic controller from Matt that allows the user to generate force wrenches using the keyboard for testing purposes. It serves as a handy example for interacting with the rest of the controller.

# Adding a controller
## How to get system state
System state is derived from the **kinematic tree**, ie tf. Use the transform between (vehicle_name)/base_link and (vehicle_name)/odom to determine the position and orientation of the vehicle. This transform is defined by the state estimation node.

## How to affect the plant
Each controller needs to publish a **WrenchStamped** message to the **/(vehicle_name)/controllers/(controller_name)/wrench** topic. The wrench must be stamped with the frame it is defined relative to, and there must exist a tf connection between that and the output frame, (vehicle_name)/base_link.

## How to add your controller to the system
You'll need to properly **namespace** your controller and add it to the "controllers" **launchfile**. You may need to create a launchfile for your controller, then include that in the "controllers" launchfile if you need to do complex setup.
The controller namespace should be $(arg model_name)/controllers, and all controller topics, services, params, etc should live in $arg model_name)/controllers/(controller_name). 

Finally, you need to **register your controller** with the control_aggregator node. This is as simple as adding an entry to the **controllers.yaml** file in the vehicle_description package for the vehicle you want the controller to run on. In addition, specify if the controller should be enabled or not. Disabling it is handy for controllers that are experimental or for testing only. All controllers can be enabled/disabled at runtime using the **controller_set_status service** from the control_aggregator node.
