from setuptools import setup

package_name = 'tauv_vehicle'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gleb Ryabtsev',
    maintainer_email='gl.ryabtsev1@gmail.com',
    description='TAUV Vehicle Drivers',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waterlinked_dvl = tauv_vehicle.waterlinked_dvl:main',
        ],
    },
)
