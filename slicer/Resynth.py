# pylint: disable=import-error, no-name-in-module, no-member, invalid-name
import time
import logging
from pathlib import Path

import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


ONE_BY_THREE_LAYOUT_ID = 502

class Resynth(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Resynth"
    self.parent.categories = ["Episurg"]
    self.parent.dependencies = []
    self.parent.contributors = [
        "Fernando Perez-Garcia (fernando.perezgarcia.17@ucl.ac.uk)",
    ]
    self.parent.helpText = """
    """
    self.parent.acknowledgementText = """
    """


class ResynthWidget(ScriptedLoadableModuleWidget):
  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.logic = ResynthLogic()

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    slicer.resynth = self
    self.makeGUI()

  def makeGUI(self):
    self.layout.addStretch()


class ResynthLogic(ScriptedLoadableModuleLogic):
  pass
