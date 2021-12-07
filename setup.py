from setuptools import setup, find_packages
from pathlib import Path
import re

entry_points = {'console_scripts': [
    'deepcompton-reco-compton-density = deepcompton.scripts.reconstruction_compton_density:main',
]}

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def get_version(prop, directory):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(directory + '/version.py').read())
    return result.group(1)


setup(
    name='deepcompton',
    version=get_version('__version__', this_directory.joinpath('deepcompton').as_posix()),
    description="Deep learning for Integral event reconstruction",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'sklearn',
        'tensorflow',
        'astropy',
        'tqdm'
    ],
    packages=find_packages(),
    scripts=[],
    tests_require=['pytest'],
    author='TBC',
    author_email='',
    url='https://github.com/vuillaut/DeepIntegralCompton',
    license='MIT',
    entry_points=entry_points,
    include_package_data=True,
    package_data={'deepcompton': ['data/*']}
)
