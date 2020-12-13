# -*- coding: utf-8 -*-

"""Top-level package for resector."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.1.2'

from . import io, parcellation
from .resector import resect
from .random_resection import RandomResection
