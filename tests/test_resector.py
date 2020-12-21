#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `resector` package."""


import unittest
import torchio as tio
from click.testing import CliRunner

from resector import resector
from resector import cli


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
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'resector.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
