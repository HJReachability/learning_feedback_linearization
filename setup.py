from os.path import join, dirname, realpath
from setuptools import setup
import sys


with open(join("spinup2", "version.py")) as version_file:
    exec(version_file.read())
setup(
    name='spinup2',
    py_modules=['spinup2'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'tensorflow',
        'tqdm'
    ],
    extras_require={'mujoco': 'mujoco-py<2.1,>=2.0'},
    description="Teaching tools for introducing people to deep RL.",
    author="Joshua Achiam",
)


    # with open(join("spinup", "version.py")) as version_file:
    #     exec(version_file.read())

    # setup(
    #     name='spinup',
    #     py_modules=['spinup'],
    #     version=__version__,#'0.1',
    #     install_requires=[
    #         'cloudpickle==0.5.2',
    #         'gym[atari,box2d,classic_control]>=0.10.8',
    #         'ipython',
    #         'joblib',
    #         'matplotlib==3.0.2',
    #         'mpi4py',
    #         'numpy',
    #         'pandas',
    #         'pytest',
    #         'psutil',
    #         'scipy',
    #         'seaborn==0.8.1',
    #         'tensorflow>=1.8.0,<2.0',
    #         'tqdm'
    #     ],
    #     extras_require={'mujoco': 'mujoco-py<2.1,>=2.0'},
    #     description="Teaching tools for introducing people to deep RL.",
    #     author="Joshua Achiam",
    # )
