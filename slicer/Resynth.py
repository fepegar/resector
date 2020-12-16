# pylint: disable=import-error, no-name-in-module, no-member, invalid-name, bad-continuation, bad-indentation
import random
from pathlib import Path

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
  ScriptedLoadableModule,
  ScriptedLoadableModuleWidget,
  ScriptedLoadableModuleLogic,
  ScriptedLoadableModuleTest,
)


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
    slicer.mrmlScene.Clear()
    self.logic = ResynthLogic()
    self.logic.installLibraries()
    self.modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    self.modelNode.CreateDefaultDisplayNodes()
    self.loadBrainMesh()
    self.makeGUI()
    slicer.resynth = self

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

  def loadBrainMesh(self):
    path = self.resourcePath('brain.vtp')
    self.brainModelNode = slicer.util.loadModel(path)
    displayNode = self.brainModelNode.GetDisplayNode()
    displayNode.SetOpacity(0.1)

  def makeGUI(self):
    self.parametersButton = ctk.ctkCollapsibleButton()
    self.parametersButton.text = 'Parameters'
    self.layout.addWidget(self.parametersButton)
    parametersLayout = qt.QFormLayout(self.parametersButton)

    self.coordinatesWidget = ctk.ctkCoordinatesWidget()
    self.coordinatesWidget.coordinates = '0,0,0'
    # self.coordinatesWidget.coordinatesChanged.connect(self.onApply)
    parametersLayout.addRow('Center: ', self.coordinatesWidget)

    self.radiusWidget = ctk.ctkCoordinatesWidget()
    self.radiusWidget.coordinates = '25,25,25'
    # self.radiusWidget.radiusChanged.connect(self.onApply)
    parametersLayout.addRow('Radii: ', self.radiusWidget)

    # slider = 'ctk'
    # if slider == 'qt':
    #   self.radiusSlider = qt.QSlider()
    #   self.radiusSlider.setOrientation(qt.Qt.Horizontal)
    #   self.radiusSlider.minimum = 10
    #   self.radiusSlider.maximum = 100
    #   self.radiusSlider.value = 50
    #   # self.radiusSlider.sliderReleased.connect(self.onApply)
    # elif slider == 'ctk':
    #   self.radiusSlider = ctk.ctkSliderWidget()
    #   self.radiusSlider.minimum = 10
    #   self.radiusSlider.maximum = 100
    #   self.radiusSlider.value = 50
    #   # self.radiusSlider.valueChanged.connect(self.onApply)
    # parametersLayout.addRow('Radius: ', self.radiusSlider)

    self.amplitudeSlider = ctk.ctkSliderWidget()
    self.amplitudeSlider.singleStep = 0.1
    self.amplitudeSlider.minimum = 0
    self.amplitudeSlider.maximum = 5
    self.amplitudeSlider.value = 0.5
    # self.amplitudeSlider.valueChanged.connect(self.onApply)
    parametersLayout.addRow('Amplitude: ', self.amplitudeSlider)

    self.smoothnessSlider = ctk.ctkSliderWidget()
    self.smoothnessSlider.singleStep = 0.1
    self.smoothnessSlider.minimum = 0.1
    self.smoothnessSlider.maximum = 10
    self.smoothnessSlider.value = 1
    # self.smoothnessSlider.valueChanged.connect(self.onApply)
    parametersLayout.addRow('Smoothness: ', self.smoothnessSlider)

    self.octavesSlider = ctk.ctkSliderWidget()
    self.octavesSlider.minimum = 1
    self.octavesSlider.maximum = 8
    self.octavesSlider.value = 4
    # self.octavesSlider.valueChanged.connect(self.onApply)
    parametersLayout.addRow('Octaves: ', self.octavesSlider)

    self.applyButton = qt.QPushButton('Apply')
    self.applyButton.clicked.connect(self.onApply)
    parametersLayout.addWidget(self.applyButton)
    self.onApply()

    self.layout.addStretch()

  def onApply(self):
    # radius = self.radiusSlider.value
    amplitude = self.amplitudeSlider.value
    center = [float(n) for n in self.coordinatesWidget.coordinates.split(',')]
    radii = [float(n) for n in self.radiusWidget.coordinates.split(',')]
    octaves = int(self.octavesSlider.value)
    smoothness = self.smoothnessSlider.value
    polyData = self.logic.getNoisySphere(
      center=center,
      radii=radii,
      noiseOffset=random.randint(0, 1000),
      octaves=octaves,
      smoothness=smoothness,
      amplitude=amplitude,
    )
    self.modelNode.SetAndObservePolyData(polyData)
    self.modelNode.GetDisplayNode().SetColor(1, 0, 1)


class ResynthLogic(ScriptedLoadableModuleLogic):

  def installLibraries(self):
    try:
      import resector
    except ImportError:
      resectorPath = Path('~/git/resector').expanduser()
      slicer.util.pip_install(f'--editable {resectorPath}')

  def getNoisySphere(self, center, radii, noiseOffset, octaves, smoothness, amplitude):
    from resector.io import get_sphere_poly_data
    from resector.mesh import get_resection_poly_data
    polyData = get_resection_poly_data(
      get_sphere_poly_data(),
      center=center,
      radii=radii,
      angles=(0, 0, 0),
      noise_offset=noiseOffset,
      octaves=octaves,
      scale=amplitude,
      smoothness=smoothness,
    )
    return polyData

  def resect(
      self,
      sample,
      resectionParams,
      ):
    from torchio import DATA, AFFINE
    from resector import resect
    from resector.io import get_sphere_poly_data, nib_to_sitk

    brain = nib_to_sitk(
      sample['image'][DATA][0],
      sample['image'][AFFINE],
    )
    hemisphere = resectionParams['hemisphere']
    gray_matter_mask = nib_to_sitk(
      sample[f'resection_gray_matter_{hemisphere}'][DATA][0],
      sample[f'resection_gray_matter_{hemisphere}'][AFFINE],
    )
    resectable_hemisphere_mask = nib_to_sitk(
      sample[f'resection_resectable_{hemisphere}'][DATA][0],
      sample[f'resection_resectable_{hemisphere}'][AFFINE],
    )
    noise_image = nib_to_sitk(
      sample['resection_noise'][DATA][0],
      sample['resection_noise'][AFFINE],
    )

    resected_brain, resection_mask, resection_center = resect(
      brain,
      gray_matter_mask,
      resectable_hemisphere_mask,
      noise_image,
      resectionParams['sigmas'],
      resectionParams['radii'],
      resectionParams['angles'],
      resectionParams['noise_offset'],
    )




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
