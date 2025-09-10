from glob import glob

from setuptools import find_packages, setup

package_name = 'tauv_common'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gleb',
    maintainer_email='gryabtse@andrew.cmu.edu',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_estimator_ekf = tauv_common.state_estimator_ekf:main',
            'depth_estimator = tauv_common.depth_estimator:main',
            'wrench_test = tauv_common.wrench_test:main',
            'thruster_manager = tauv_common.thruster_manager:main',
            'commander = tauv_common.commander:main',
            'controller = tauv_common.controller:main',
            'commander_example = tauv_common.commander_example:main',
        ],
    },
)
