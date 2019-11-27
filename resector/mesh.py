import sys
import time
from pathlib import Path
from subprocess import call, DEVNULL

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
from noise import pnoise3, snoise3


def get_sphere_poly_data(center, radius, theta=64, phi=64):
    # TODO: apply trsf to sphere to make it an ellipsoid
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(center)
    sphere_source.SetRadius(radius)
    if theta is not None:
        sphere_source.SetThetaResolution(theta)
    if phi is not None:
        sphere_source.SetPhiResolution(phi)
    sphere_source.Update()
    return sphere_source.GetOutput()
    # return get_ellipsoid_poly_data(center, radii_ratio, (0, 0, 0))


def get_ellipsoid_poly_data(
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

    parametricSource = vtk.vtkParametricFunctionSource()
    parametricSource.SetParametricFunction(ellipsoid)
    parametricSource.Update()

    transform = vtk.vtkTransform()
    transform.Translate(center)
    x_angle, y_angle, z_angle = angles
    transform.RotateX(x_angle)
    transform.RotateY(y_angle)
    transform.RotateZ(z_angle)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(transform)
    tf.SetInputData(parametricSource.GetOutput())
    tf.Update()

    poly_data = tf.GetOutput()
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


def mesh_to_volume(reference_path, model_path, result_path, verbose=False):
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


def write_poly_data(poly_data, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_data)
    writer.SetFileName(path)
    return writer.Write()
