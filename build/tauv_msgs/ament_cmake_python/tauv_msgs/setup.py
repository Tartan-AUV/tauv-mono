from setuptools import find_packages
from setuptools import setup

setup(
    name='tauv_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('tauv_msgs', 'tauv_msgs.*')),
)
