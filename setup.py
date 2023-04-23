#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click>=7.0',
    'noise',
    'torchio',
    'SimpleITK',
    'scikit-image',
    'scipy',
    'vtk',
]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Fernando Perez-Garcia",
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Algorithm to simulate resections osurgery on brain MRI scans.",
    entry_points={
        'console_scripts': [
            'resect=resector.cli.resect:main',
            'create-noise-volume=resector.cli.create_noise_volume:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='resector',
    name='resector',
    packages=find_packages(include=['resector', 'resector.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fepegar/resector',
    version='0.2.10',
    zip_safe=False,
)
