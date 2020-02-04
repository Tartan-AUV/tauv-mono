# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['thruster_managers',
              'thruster_managers.models',
              'control.cascaded_pids',
              'control',
              'teleop'],
    package_dir={'': 'src'},
    requires=['rospy'],
    scripts=['scripts/thruster_allocator',
             'scripts/keyboard_controller',
             'scripts/acceleration_controller',
             'scripts/velocity_controller',
             'scripts/position_controller',
             'scripts/teleop']
)

setup(**setup_args)
