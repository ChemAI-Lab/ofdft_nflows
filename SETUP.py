import os
from setuptools import find_packages
from setuptools import setup

setup(
    name='pipjax',
    packages=find_packages(),
    version='0.1',
    description='Leveraging Normalizing Flows for Orbital-Free Density Functional Theory',
    author='Rodrigo. A. Vargas-Hernandez',
    install_requires=[
        'jax>0.4.14',
        'jaxlib>0.4.14',
        'numpy>=1.18.0',
        'chex>=0.1.7',
        'typing_extensions>=4.8.0',
        'jaxtyping',
        'flax',
        'pytest>=7.4.3',
        'optax>0.1.7',
        'orbax-checkpoint>0.4.4',
    ],
    python_requires='==3.9.6',
    keywords="machine learning, normalizing flows, orbital free density functional theory",
)