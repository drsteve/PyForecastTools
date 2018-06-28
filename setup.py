#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='PyForecastTools',
      version='1.0.1',
      description='Model validation and forecast verification tools',
      author='Steve Morley',
      author_email='smorley@lanl.gov',
      license='BSD License',
      url='https://github.com/drsteve/PyForecastTools',
      install_requires=['numpy'],
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(exclude=['tests']),
      classifiers=['Development Status :: 4 - Beta',
                   'License :: OSI Approved :: BSD License',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Atmospheric Science',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   ],
     test_suite='test_verify.py'
     )
