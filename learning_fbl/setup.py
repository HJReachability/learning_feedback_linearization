"""
Setup Packages
Author: Valmik
"""
from setuptools import setup

requirements = []

setup(name='learning_fbl',
      version='0.1.0',
      description='Base package for learning feedback linearization',
      author='Valmik Prabhu',
      author_email='valmik@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['fbl_core'],
      install_requires=requirements,
      test_suite='test'
     )