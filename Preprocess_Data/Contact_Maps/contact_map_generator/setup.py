from setuptools import setup, find_packages

setup(
    name='contact_map_generator',
    version='0.1.0',
    author='Honglue Shi',
    author_email='marenatrinidad@berkeley.edu',
    description='Generate contact maps for GNN training',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.26.3',
        'pandas>=2.1.4',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.10',
)

