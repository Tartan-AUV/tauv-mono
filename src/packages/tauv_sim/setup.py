from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['thrusters_sim', 'xsens_imu_sim', 'teledyne_dvl_sim'],
    package_dir={'': 'src'},
    requires=['rospy'],
    scripts=['scripts/thrusters_sim', 'scripts/xsens_imu_sim', 'scripts/teledyne_dvl_sim']
)

setup(**setup_args)
