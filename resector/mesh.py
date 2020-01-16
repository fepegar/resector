import sys
import time
import shutil
import warnings
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


def get_resection_poly_data(
        poly_data,
        center,
        radii,
        angles,
        noise_offset=None,
        octaves=4,
        scale=0.5,
        deepcopy=True,
        ):
    if deepcopy:
        new_poly_data = vtk.vtkPolyData()
        new_poly_data.DeepCopy(poly_data)
        poly_data = new_poly_data
    noise_offset = 1000 if noise_offset is None else noise_offset
    poly_data = add_noise_to_sphere(
        poly_data,
        octaves=octaves,
        offset=noise_offset,
    )
    poly_data = center_poly_data(poly_data)
    poly_data = transform_poly_data(poly_data, center, radii, angles)
    poly_data = compute_normals(poly_data)
    return poly_data


def add_noise_to_sphere(poly_data, octaves, offset=0, scale=0.5):
    """
    Expects sphere with radius 1 centered at the origin
    """
    wrap_data_object = dsa.WrapDataObject(poly_data)
    points = wrap_data_object.Points
    normals = wrap_data_object.PointData['Normals']

    points_with_noise = []
    for point, normal in zip(points, normals):
        offset_point = point + offset
        noise = scale * snoise3(*offset_point, octaves=octaves)
        point_with_noise = point + noise * normal
        points_with_noise.append(point_with_noise)
    points_with_noise = np.array(points_with_noise)

    vertices = vtk.vtkPoints()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        points_with_noise_vtk = numpy_to_vtk(points_with_noise)
    vertices.SetData(points_with_noise_vtk)
    poly_data.SetPoints(vertices)
    return poly_data


def center_poly_data(poly_data):
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(poly_data)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())

    transform = vtk.vtkTransform()
    transform.Translate(-center)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(poly_data)
    transform_filter.Update()

    poly_data = transform_filter.GetOutput()
    return poly_data


def transform_poly_data(poly_data, center, radii, angles):
    transform = vtk.vtkTransform()
    transform.Translate(center)
    x_angle, y_angle, z_angle = angles  # there must be a better way
    transform.RotateX(x_angle)
    transform.RotateY(y_angle)
    transform.RotateZ(z_angle)
    transform.Scale(*radii)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(poly_data)
    transform_filter.Update()

    poly_data = transform_filter.GetOutput()
    return poly_data


def compute_normals(poly_data):
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.SetComputePointNormals(True)
    normal_filter.SetComputeCellNormals(True)
    normal_filter.SplittingOff()
    normal_filter.SetInputData(poly_data)
    normal_filter.ConsistencyOn()
    normal_filter.Update()
    poly_data = normal_filter.GetOutput()
    return poly_data


def mesh_to_volume(poly_data, reference_path):
    """
    ASSUME INPUT IN RAS
    TODO: stop reading and writing so much stuff
    Write to buffer? Bytes? Investigate this
    """
    def check_header(nifti_image):
        orientation = ''.join(nib.aff2axcodes(nifti_image.affine))
        is_ras = orientation == 'RAS'
        if not is_ras:
            message = (
                'RAS orientation expected.'
                f' Detected orientation: {orientation}'
            )
            raise Exception(message)
        qform_code = nifti_image.header['qform_code']
        if qform_code == 0:
            raise Exception(f'qform code for {reference_path} is 0')

    nii = nib.load(str(reference_path))
    check_header(nii)
    image_stencil_array = np.ones(nii.shape, dtype=np.uint8)
    image_stencil_nii = nib.Nifti1Image(image_stencil_array, nii.affine)  # nii.get_qform())
    image_stencil_nii.header['qform_code'] = 1
    image_stencil_nii.header['sform_code'] = 0

    with NamedTemporaryFile(suffix='.nii') as f:
        stencil_path = f.name
        image_stencil_nii.to_filename(stencil_path)
        image_stencil_reader = vtk.vtkNIFTIImageReader()
        image_stencil_reader.SetFileName(stencil_path)
        image_stencil_reader.Update()
        pass

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
    array = array.reshape(nii.shape, order='F')  # as order='C' didn't work
    array = check_qfac(nii, array)

    num_voxels = array.sum()
    if num_voxels == 0:
        import warnings
        warnings.warn(f'Empty stencil mask for reference {reference_path}')

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
