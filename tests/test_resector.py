#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `resector` package."""


import unittest
import torchio as tio
from click.testing import CliRunner


class TestResector(unittest.TestCase):
    """Tests for `resector` package."""

    def setUp(self):
        self.subject = tio.datasets.FPG()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
