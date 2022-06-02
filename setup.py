from setuptools import setup

setup(
    name='nnfflib',
    version='0.0.1',
    description='A Neural Network Force Field package',
    author='Maarten Cools-Ceuppens',
    packages=['nnfflib'],
    include_package_data=True,
    package_data = {'': ['*.so']}
    )

