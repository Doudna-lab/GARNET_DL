# The setup.py file is used as the build script for setuptools. Setuptools is a
# package that allows you to easily build and distribute Python distributions.

import setuptools
import os
from version import version as this_version

# write version to PlotMAPQ directory so it can be accessed by the command line
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'struct2seq', '_version.py'),
          'wt') as fversion:
    fversion.write('__version__ = "' + this_version + '"')

# Define required packages. Alternatively, these could be defined in a separate
# file and read in here.
REQUIRED_PACKAGES = []

# Read in the project description. We define this in the README file.
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="struct2seq",  # name of project
    install_requires=REQUIRED_PACKAGES,  # all requirements used by this package
    version=this_version,  # project version, read from version.py
    author="Hunter Nisonoff",  # Author, shown on PyPI
    #scripts = ['bin/plot-mapq'],                                # command line scripts installed
    author_email="hunter_nisonoff@berkeley.edu",  # Author email
    description="TODO",  # Short description of project
    long_description=long_description,  # Long description, shown on PyPI
    long_description_content_type=
    "text/markdown",  # Content type. Here, we used a markdown file.
    #url="https://github.com/akmorrow13/CompBIO_Seminar_2020",   # github path
    packages=setuptools.find_packages(
    ),  # automatically finds packages in the current directory. You can also explictly list them.
    classifiers=
    [  # Classifiers give pip metadata about your project. See https://pypi.org/classifiers/ for a list of available classifiers.
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # python version requirement
)
