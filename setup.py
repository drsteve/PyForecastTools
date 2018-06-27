#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='PyForecastTools',
      version='1.0',
      description='Model validation and forecast verification tools',
      author='Steve Morley',
      author_email='smorley@lanl.gov',
      license='BSD License',
      url='https://drsteve.github.io/PyForecastTools',
      install_requires=['numpy'],
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
