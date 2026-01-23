from glob import glob

from setuptools import setup

package_name = 'tauv_mission'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gleb',
    maintainer_email='gryabtse@andrew.cmu.edu',
    description='Mission-level tools and teleoperation CLI for TartanAUV vehicles',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop = tauv_mission.teleop:main',
        ],
    },
)
