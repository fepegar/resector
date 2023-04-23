
import gzip
import shutil
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile

import vtk
import numpy as np
import torchio as tio
import nibabel as nib
import SimpleITK as sitk


CHECK_QFAC = False
debug_dir = None
debug_num_files = 0


def read_itk(image_path):
    image_path = str(image_path)
    if CHECK_QFAC:
        check_qfac(image_path)
    image = sitk.ReadImage(image_path)
    image += 0  # https://discourse.itk.org/t/simpleitk-writing-nifti-with-invalid-header/2498/4
    return image


def check_qfac(image_path):
    image_path = Path(image_path)
    with NamedTemporaryFile(suffix='.nii') as f:
        if image_path.suffix == '.gz':
            with gzip.open(image_path, 'rb') as f_in:
                with open(f.name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            image_path = f.name
        with open(image_path, 'rb') as f:
            fmt = 8 * 'f'
            size = struct.calcsize(fmt)
            f.seek(76)
            chunk = f.read(size)
            pixdim = struct.unpack(fmt, chunk)
    qfac = pixdim[0]
    if qfac not in (-1, 1):
        raise ValueError(f'qfac is {qfac} in {image_path}')


def write(image, image_path, set_sform_code_zero=True):
    image_path = str(image_path)
    sitk.WriteImage(image, image_path)
    if CHECK_QFAC:
        check_qfac(image_path)
    if set_sform_code_zero:
        nii = nib.load(image_path)
        data = np.asanyarray(nii.dataobj)
        if isinstance(data, np.memmap):
            data = np.array(data)
        nii = nib.Nifti1Image(data, nii.affine)
        nii.header['qform_code'] = 1
        nii.header['sform_code'] = 0
        nii.to_filename(image_path)
    if CHECK_QFAC:
        check_qfac(image_path)


def get_sphere_poly_data():
    resources_dir = Path(__file__).parent / 'resources'
    mesh_path = resources_dir / 'geodesic_polyhedron.vtp'
    if not mesh_path.is_file():
        raise FileNotFoundError(f'{mesh_path} does not exist')
    poly_data = read_poly_data(mesh_path)
    if poly_data.GetNumberOfPoints() == 0:
        message = (
            f'Error reading sphere poly data from {mesh_path}. Contents:'
            f'\n{mesh_path.read_text()}'
        )
        raise FileNotFoundError(message)
    return poly_data


def read_poly_data(path, flip=False):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly_data = reader.GetOutput()
    if flip:
        from .mesh import flipxy
        poly_data = flipxy(poly_data)
    return poly_data


def write_poly_data(poly_data, path, flip=False):
    if flip:
        from .mesh import flipxy
        poly_data = flipxy(poly_data)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_data)
    writer.SetFileName(str(path))
    return writer.Write()


def save_debug(x):
    global debug_dir
    global debug_num_files
    if debug_dir is None:
        return
    debug_dir.mkdir(exist_ok=True, parents=True)
    # https://stackoverflow.com/a/10361425/3956024
    import traceback
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    filename = Path(filename).stem
    varname = code.split('(')[-1][:-1]
    out_stem = f'{debug_num_files:02d}_{filename}_{lineno}_{function_name}_{varname}'
    if isinstance(x, vtk.vtkPolyData):
        path = debug_dir / f'{out_stem}.vtp'
        write_poly_data(x, path, flip=True)
    elif isinstance(x, sitk.Image):
        path = debug_dir / f'{out_stem}.nii.gz'
        write(x, path)
    else:
        raise TypeError(f'Type not understood: {type(x)}')
    debug_num_files += 1


def nib_to_sitk(array: np.ndarray, affine: np.ndarray):
    assert array.ndim == 3
    array = array[np.newaxis]
    return tio.io.nib_to_sitk(array, affine)
