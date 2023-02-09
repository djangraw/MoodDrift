
from setuptools import setup, find_packages


setup(
    name='MoodDrift',
    version='0.0.1',
    author='David Jangraw',
    author_email='djangraw@gmail.com',
    description='scripts and functions for MoodDrift paper',
    url='https://github.com/djangraw/MoodDrift',
    py_modules=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX",
    ]
)