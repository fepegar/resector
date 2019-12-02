import sys
import time
import shutil
from pathlib import Path
from subprocess import call, DEVNULL
from tempfile import NamedTemporaryFile

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
import nibabel as nib
from noise import pnoise3, snoise3

from .io import nib_to_sitk


def get_ellipsoid_poly_data(center, radii, angles, theta=64, phi=64):
    """
    Radii can have length 1 or 3
    Angles are in degrees
    """
    try:
        iter(radii)
    except TypeError:
        radii = 3 * (radii,)
    sphere_source = vtk.vtkSphereSource()
    if theta is not None:
        sphere_source.SetThetaResolution(theta)
    if phi is not None:
        sphere_source.SetPhiResolution(phi)
    sphere_source.Update()

    transform = vtk.vtkTransform()
    transform.Translate(center)
    x_angle, y_angle, z_angle = angles  # there must be a better way
    transform.RotateX(x_angle)
    transform.RotateY(y_angle)
    transform.RotateZ(z_angle)
    transform.Scale(*radii)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(sphere_source.GetOutputPort())
    transform_filter.Update()

    poly_data = transform_filter.GetOutput()
    return poly_data


def get_ellipsoid_poly_data_(
        center,
        radius,
        radii_ratio,
        angles,
        ):
    """
    Input angles are in degrees
    """
    ellipsoid = vtk.vtkParametricEllipsoid()
    a = radius
    b = a * radii_ratio
    c = radius**3 / (a * b)  # a * b * c = r**3
    ellipsoid.SetXRadius(a)
    ellipsoid.SetYRadius(b)
    ellipsoid.SetZRadius(c)

    parametric_source = vtk.vtkParametricFunctionSource()
    parametric_source.SetParametricFunction(ellipsoid)
    parametric_source.Update()

    transform = vtk.vtkTransform()
    transform.Translate(center)
    x_angle, y_angle, z_angle = angles
    transform.RotateX(x_angle)
    transform.RotateY(y_angle)
    transform.RotateZ(z_angle)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(parametric_source.GetOutput())
    transform_filter.Update()

    poly_data = transform_filter.GetOutput()
    return poly_data


def add_noise_to_poly_data(
        poly_data,
        radius,
        noise_type=None,
        octaves=None,
        persistence=None,
        smoothness=None,
        noise_amplitude_radius_ratio=None,
        verbose=False,
    ):
    if verbose:
        start = time.time()
    noise_type = 'perlin' if noise_type is None else noise_type
    assert noise_type in 'perlin', 'simplex'
    octaves = 4 if octaves is None else octaves  # make random?
    persistence = 0.7 if persistence is None else persistence  # make random?
    smoothness = 10 if smoothness is None else smoothness  # make random?
    if noise_amplitude_radius_ratio is None:  # make random?
        noise_amplitude_radius_ratio = 0.5

    wrap_data_object = dsa.WrapDataObject(poly_data)
    points = wrap_data_object.Points
    normals = wrap_data_object.PointData['Normals']

    noise_amplitude = radius * noise_amplitude_radius_ratio
    if noise_type == 'perlin':
        function = pnoise3
    elif noise_type == 'simplex':
        function = snoise3

    points_with_noise = []
    for point, normal in zip(points, normals):
        x, y, z = point / smoothness
        noise = function(x, y, z, octaves=octaves)#, persistence=persistence)  # add random offset?
        perturbance = noise_amplitude * noise
        point_with_noise = point + perturbance * normal
        points_with_noise.append(point_with_noise)
    points_with_noise = np.array(points_with_noise)

    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(points_with_noise))
    poly_data.SetPoints(vertices)

    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.SetComputePointNormals(True)
    normal_filter.SetComputeCellNormals(True)
    normal_filter.SplittingOff()
    normal_filter.SetInputData(poly_data)
    normal_filter.ConsistencyOn()
    normal_filter.Update()
    poly_data = normal_filter.GetOutput()
    if verbose:
        duration = time.time() - start
        print(f'add_noise_to_poly_data: {duration:.1f} seconds')
    return poly_data


def get_slicer_paths(version='4.11'):
    platform = sys.platform
    if platform == 'darwin':  # macOS
        slicer_dir = Path('/Applications/Slicer.app/Contents')
        slicer_bin_path = slicer_dir / 'MacOS' / 'Slicer'
    elif platform == 'linux':
        slicer_dir = Path('~/opt/Slicer/Nightly').expanduser()
        slicer_bin_path = slicer_dir / 'Slicer'
    lib_dir = slicer_dir / 'lib' / f'Slicer-{version}'
    modules_dir = lib_dir / 'cli-modules'
    module_name = 'ModelToLabelMap'
    slicer_module_path = modules_dir / module_name

    import os
    environ_dict = dict(os.environ)
    if 'LD_LIBRARY_PATH' not in environ_dict:
        environ_dict['LD_LIBRARY_PATH'] = ''
    environ_dict['LD_LIBRARY_PATH'] += f':{modules_dir}'
    environ_dict['LD_LIBRARY_PATH'] += f':{lib_dir}'
    os.environ.clear()
    os.environ.update(environ_dict)

    return slicer_bin_path, slicer_module_path


def mesh_to_volume_slicer(reference_path, model_path, result_path, verbose=False):
    # There must be something already in result_path so that
    # Slicer can load the vtkMRMLVolumeNode? So many ugly hacks :(
    shutil.copy2(reference_path, result_path)
    # TODO: check if the previous copy is still needed

    if verbose:
        start = time.time()
    _, slicer_module_path = get_slicer_paths()
    command = (
        slicer_module_path,
        reference_path,
        model_path,
        result_path,
    )
    command = [str(a) for a in command]
    call(command, stderr=DEVNULL, stdout=DEVNULL)
    if verbose:
        duration = time.time() - start
        print(f'mesh_to_volume: {duration:.1f} seconds')


def mesh_to_volume(poly_data, reference_path):
    """
    TODO: stop reading and writing so much stuff
    Write to buffer? Bytes? Investigate this
    """
    nii = nib.load(str(reference_path))
    image_stencil_array = np.ones(nii.shape, dtype=np.uint8)
    image_stencil_nii = nib.Nifti1Image(image_stencil_array, nii.get_qform())
    image_stencil_nii.header['qform_code'] = 1
    image_stencil_nii.header['sform_code'] = 0

    with NamedTemporaryFile(suffix='.nii') as f:
        stencil_path = f.name
        image_stencil_nii.to_filename(stencil_path)
        image_stencil_reader = vtk.vtkNIFTIImageReader()
        image_stencil_reader.SetFileName(stencil_path)
        image_stencil_reader.Update()

    image_stencil = image_stencil_reader.GetOutput()
    xyz_to_ijk = image_stencil_reader.GetQFormMatrix()
    if xyz_to_ijk is None:
        import warnings
        warnings.warn('No qform found. Using sform')
        xyz_to_ijk = image_stencil_reader.GetSFormMatrix()
    xyz_to_ijk.Invert()

    transform = vtk.vtkTransform()
    transform.SetMatrix(xyz_to_ijk)

    transform_poly_data = vtk.vtkTransformPolyDataFilter()
    transform_poly_data.SetTransform(transform)
    transform_poly_data.SetInputData(poly_data)
    transform_poly_data.Update()
    pd_ijk = transform_poly_data.GetOutput()

    polyDataToImageStencil = vtk.vtkPolyDataToImageStencil()
    polyDataToImageStencil.SetInputData(pd_ijk)
    polyDataToImageStencil.SetOutputSpacing(image_stencil.GetSpacing())
    polyDataToImageStencil.SetOutputOrigin(image_stencil.GetOrigin())
    polyDataToImageStencil.SetOutputWholeExtent(image_stencil.GetExtent())
    polyDataToImageStencil.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image_stencil)
    stencil.SetStencilData(polyDataToImageStencil.GetOutput())
    stencil.SetBackgroundValue(0)
    stencil.Update()

    image_output = stencil.GetOutput()

    data_object = dsa.WrapDataObject(image_output)
    array = data_object.PointData['NIFTI']
    array = array.reshape(nii.shape, order='F')  # C didn't work :)
    array = check_qfac(nii, array)

    output_image = nib_to_sitk(array, nii.affine)
    return output_image


def check_qfac(nifti, array):
    """
    See https://vtk.org/pipermail/vtk-developers/2016-November/034479.html
    """
    qfac = nifti.header['pixdim'][0]
    if qfac not in (-1, 1):
        raise ValueError(f'Unknown qfac value: {qfac}')
    elif qfac == -1:
        array = array[..., ::-1]
    return array


def write_poly_data(poly_data, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_data)
    writer.SetFileName(path)
    return writer.Write()
