from setuptools import find_packages, setup

package_name = 'camera_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/camera_pkg']),
        ('share/camera_pkg', ['package.xml']),
        ('share/camera_pkg/launch', ['launch/lucid_launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tauv',
    maintainer_email='tauv@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lucid = camera_pkg.lucid:main'
        ],
    },
)
