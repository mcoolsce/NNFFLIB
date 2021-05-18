from setuptools import setup

setup(
    name='nnff',
    version='0.0.1',
    description='A Neural Network Force Field package',
    author='Maarten Cools-Ceuppens',
    packages=['nnff'],
    include_package_data=True,
    package_data = {'nnff/cell_list_op.so': ['nnff/cell_list_op.so']}
    )

