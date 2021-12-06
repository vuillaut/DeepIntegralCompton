import re
from setuptools import setup, find_packages
from pathlib import Path


entry_points = {'console_scripts': []}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
__version__ = open(Path(this_directory).joinpath('deepcompton/version.py'))

setup(
    name='deepcompton',
    version=__version__,
    description="Deep learning for Integral event reconstruction",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
    ],
    packages=find_packages(),
    scripts=[],
    tests_require=['pytest'],
    author='TBC',
    author_email='',
    url='https://github.com/vuillaut/DeepIntegralCompton',
    license='MIT',
    entry_points=entry_points,
    package_data={}
)