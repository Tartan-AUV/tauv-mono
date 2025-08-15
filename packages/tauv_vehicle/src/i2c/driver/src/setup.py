from setuptools import setup, find_packages

setup(
    name='osprey_i2c_driver',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'smbus2',
    ],
    python_requires='>=3.6',
)
