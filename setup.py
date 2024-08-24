"""Setup script for GPNN package."""

from setuptools import setup, find_packages

def read_requirements():
    """Parse requirements from requirements.txt."""
    with open('requirements.txt', encoding='UTF-8') as req:
        return req.read().splitlines()

setup(
    name='gpnn',
    version='0.0.1',
    author='Jake Vikoren',
    author_email='jake.vikoren@genmat.xyz',
    description='An ML Model for the prediction of charge density.',
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GenerativeMaterials/GPNN',
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.7',
)
