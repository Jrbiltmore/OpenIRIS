
from setuptools import setup, find_packages

setup(
    name='OpenIRIS',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'pytest',
    ],
)
