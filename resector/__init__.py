# -*- coding: utf-8 -*-

"""Top-level package for resector."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.2.10'

from .timer import timer
from .resector import resect
from . import io, parcellation, image
from .random_resection import RandomResection

__all__ = [
    'image',
    'io',
    'parcellation',
    'resect',
    'timer',
    'RandomResection',
]
