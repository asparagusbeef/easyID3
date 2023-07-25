from setuptools import setup, find_packages

setup(
    name='easyID3',
    version='0.1.3',
    packages=find_packages(),
    description='Implementation of the original ID3 algorithm in Python.',
    author='Jonathan',
    url='https://github.com/asparagusbeef/easyID3',
    install_requires=[
        'numpy',
        'pandas'
    ]
)
