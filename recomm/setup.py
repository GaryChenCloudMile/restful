from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud>=0.32.0', 'google-api-python-client>=1.6.5']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        '': ['*.yaml']
    },
    description='Recommendation trainer application'
)
