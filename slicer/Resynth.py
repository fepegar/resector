# pylint: disable=import-error, no-name-in-module, no-member, invalid-name
import time
import logging
from pathlib import Path

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
  ScriptedLoadableModuleTest,
)

import numpy as np
from noise import pnoise3


RESOLUTION = 64


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
    slicer.mrmlScene.Clear()
    slicer.resynth = self
    self.loadBrainMesh()
    self.sphereSource = self.logic.getSphereSource()
    self.normalFilter = self.getNormalFilter()
    self.modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    self.modelNode.CreateDefaultDisplayNodes()
    self.makeGUI()

  def loadBrainMesh(self):
    path = self.resourcePath('brain.vtp')
    self.brainModelNode = slicer.util.loadModel(path)
    displayNode = self.brainModelNode.GetDisplayNode()
    displayNode.SetOpacity(0.1)

  def getNormalFilter(self):
    normalFilter = vtk.vtkPolyDataNormals()
    normalFilter.AutoOrientNormalsOn()
    normalFilter.SetComputePointNormals(True)
    normalFilter.SetComputeCellNormals(True)
    normalFilter.SplittingOff()
    normalFilter.ConsistencyOn()
    return normalFilter

  def makeGUI(self):
    self.parametersButton = ctk.ctkCollapsibleButton()
    self.parametersButton.text = 'Parameters'
    self.layout.addWidget(self.parametersButton)
    parametersLayout = qt.QFormLayout(self.parametersButton)

    self.coordinatesWidget = ctk.ctkCoordinatesWidget()
    self.coordinatesWidget.coordinates = '10,20,30'
    self.coordinatesWidget.coordinatesChanged.connect(self.onApply)
    parametersLayout.addRow('Center: ', self.coordinatesWidget)

    slider = 'ctk'
    if slider == 'qt':
      self.radiusSlider = qt.QSlider()
      self.radiusSlider.setOrientation(qt.Qt.Horizontal)
      self.radiusSlider.minimum = 10
      self.radiusSlider.maximum = 50
      self.radiusSlider.value = 30
      self.radiusSlider.sliderReleased.connect(self.onApply)
    elif slider == 'ctk':
      self.radiusSlider = ctk.ctkSliderWidget()
      self.radiusSlider.minimum = 10
      self.radiusSlider.maximum = 50
      self.radiusSlider.value = 30
      self.radiusSlider.valueChanged.connect(self.onApply)
    parametersLayout.addRow('Radius: ', self.radiusSlider)

    self.amplitudeSlider = ctk.ctkSliderWidget()
    self.amplitudeSlider.singleStep = 0.1
    self.amplitudeSlider.minimum = 0
    self.amplitudeSlider.maximum = 2
    self.amplitudeSlider.value = 0
    self.amplitudeSlider.valueChanged.connect(self.onApply)
    parametersLayout.addRow('Amplitude: ', self.amplitudeSlider)

    self.applyButton = qt.QPushButton('Apply')
    self.applyButton.clicked.connect(self.onApply)
    parametersLayout.addWidget(self.applyButton)
    self.onApply()

    self.layout.addStretch()

  def onApply(self):
    radius = self.radiusSlider.value
    amplitude = self.amplitudeSlider.value
    center = [float(n) for n in self.coordinatesWidget.coordinates.split(',')]

    sphereSource = self.logic.getSphereSource()
    sphereSource.SetRadius(40)
    sphereSource.SetCenter(center)
    sphereSource.Update()
    self.polyData = sphereSource.GetOutput()

    # wrapDataObject = dsa.WrapDataObject(self.polyData)
    # points = wrapDataObject.Points
    # normals = wrapDataObject.PointData['Normals']
    # noiseAmplitude = radius * amplitude

    # pointsWithNoise = []
    # for point, normal in zip(points, normals):
    #   x, y, z = point / 100
    #   noise = pnoise3(x, y, z, octaves=4)
    #   # , persistence=persistence)  # add random offset?
    #   perturbance = noiseAmplitude * noise
    #   pointWithNoise = point + perturbance * normal
    #   pointsWithNoise.append(pointWithNoise)
    # pointsWithNoise = np.array(pointsWithNoise)

    # vertices = vtk.vtkPoints()
    # vertices.SetData(numpy_to_vtk(pointsWithNoise))
    # self.polyData.SetPoints(vertices)

    normalFilter = self.getNormalFilter()
    normalFilter.SetInputData(self.polyData)
    normalFilter.Update()

    # transform = vtk.vtkTransform()
    # transform.Scale(3 * (radius,))
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetInputData(normalFilter.GetOutput())
    # transformFilter.SetTransform(transform)
    transformFilter.Update()

    self.modelNode.SetAndObserveMesh(transformFilter.GetOutput())



class ResynthLogic(ScriptedLoadableModuleLogic):
  def getSphereSource(self):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetPhiResolution(RESOLUTION)
    sphereSource.SetThetaResolution(RESOLUTION)
    sphereSource.Update()
    return sphereSource


# # pylint: disable=import-error, no-name-in-module, no-member, invalid-name


# class Resynth(ScriptedLoadableModule):
#   def __init__(self, parent):
#   ScriptedLoadableModule.__init__(self, parent)
#   self.parent.title = "Resynth"
#   self.parent.categories = ["Episurg"]
#   self.parent.dependencies = []
#   self.parent.contributors = [
#     "Fernando Perez-Garcia (fernando.perezgarcia.17@ucl.ac.uk)",
#   ]
#   self.parent.helpText = """
#   """
#   self.parent.acknowledgementText = """
#   """


# class ResynthWidget(ScriptedLoadableModuleWidget):
#   def __init__(self, parent):
#   ScriptedLoadableModuleWidget.__init__(self, parent)
#   self.logic = ResynthLogic()

#   def setup(self):
#   ScriptedLoadableModuleWidget.setup(self)
#   slicer.resynth = self
#   self.sphereSource = self.logic.getSphereSource()
#   self.normalFilter = self.getNormalFilter()
#   self.normalFilter.SetInputData(self.sphereSource.GetOutput())
#   self.polyData = self.normalFilter.GetOutput()
#   self.modelNode = slicer.modules.models.logic().AddModel(self.polyData)
#   self.modelNode.CreateDefaultDisplayNodes()
#   self.makeGUI()

#   def getNormalFilter(self):
#   normalFilter = vtk.vtkPolyDataNormals()
#   normalFilter.AutoOrientNormalsOn()
#   normalFilter.SetComputePointNormals(True)
#   normalFilter.SetComputeCellNormals(True)
#   normalFilter.SplittingOff()
#   normalFilter.ConsistencyOn()
#   return normalFilter

#   def makeGUI(self):
#   self.parametersButton = ctk.ctkCollapsibleButton()
#   self.parametersButton.text = 'Parameters'
#   self.layout.addWidget(self.parametersButton)
#   parametersLayout = qt.QFormLayout(self.parametersButton)

#   self.radiusSlider = ctk.ctkSliderWidget()
#   self.radiusSlider.minimum = 10
#   self.radiusSlider.maximum = 100
#   self.radiusSlider.value = 30
#   self.radiusSlider.valueChanged.connect(self.onRadiusSlider)
#   self.onRadiusSlider()
#   parametersLayout.addRow('Radius: ', self.radiusSlider)

#   self.layout.addStretch()

#   def onRadiusSlider(self):
#   radius = self.radiusSlider.value
#   self.sphereSource.SetRadius(radius)
#   self.normalFilter.Update()

#   def onNoiseWidget(self):
#   octaves = 4
#   radius = self.radiusSlider.value
#   noiseAmplitudeRadiusRatio = 1

#   wrapDataObject = dsa.WrapDataObject(self.polyData)
#   points = wrapDataObject.Points
#   normals = wrapDataObject.PointData['Normals']
#   noiseAmplitude = radius * noiseAmplitudeRadiusRatio

#   pointsWithNoise = []
#   for point, normal in zip(points, normals):
#     x, y, z = point / smoothness
#     noise = function(x, y, z, octaves=octaves)
#     # , persistence=persistence)  # add random offset?
#     perturbance = noiseAmplitude * noise
#     pointWithNoise = point + perturbance * normal
#     pointsWithNoise.append(pointWithNoise)
#   pointsWithNoise = np.array(pointsWithNoise)

#   vertices = vtk.vtkPoints()
#   vertices.SetData(numpy_to_vtk(pointsWithNoise))
#   self.polyData.SetPoints(vertices)
#   self.normalFilter.Update()


# class ResynthLogic(ScriptedLoadableModuleLogic):
#   def getSphereSource(self):
#   sphereSource = vtk.vtkSphereSource()
#   sphereSource.Update()
#   return sphereSource
