from pathlib import Path
import vtk
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def read(image_path):
    return sitk.ReadImage(str(image_path))


def get_qfac(image_path):
    from struct import calcsize, unpack
    from tempfile import NamedTemporaryFile
    image_path = Path(image_path)
    with NamedTemporaryFile(suffix='.nii') as f:
        if image_path.suffix == '.gz':
            from subprocess import call
            call('gunzip', '-c', str(image_path), '>', f.name)
        image_path = f.name
        with open(image_path, 'rb') as f:
            fmt = 8 * 'f'
            size = struct.calcsize(fmt)
            f.seek(76)
            chunk = f.read(size)
    pixdim = struct.unpack(fmt, chunk)
    qfac = pixdim[0]
    return qfac


def write(image, image_path, set_sform_code_zero=True):
    image_path = str(image_path)
    sitk.WriteImage(image, image_path)
    if set_sform_code_zero:
        nii = nib.load(image_path)
        data = nii.get_data()
        if isinstance(data, np.memmap):
            data = np.array(data)
        nii = nib.Nifti1Image(data, nii.affine)
        nii.header['qform_code'] = 1
        nii.header['sform_code'] = 0
        nii.to_filename(image_path)


def nib_to_sitk(array, affine):
    """
    This actually seems faster than .parcellation.get_image_from_reference
    """
    import nibabel as nib
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix='.nii') as f:
        nib.Nifti1Image(array, affine).to_filename(f.name)
        image = sitk.ReadImage(f.name)
    return image


def get_sphere_poly_data():
    resources_dir = Path(__file__).parent.parent / 'resources'
    mesh_path = resources_dir / 'geodesic_polyhedron.vtp'
    if not mesh_path.is_file():
        raise FileNotFoundError(f'{mesh_path} does not exist')
    return read_poly_data(mesh_path)


def read_poly_data(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def write_poly_data(poly_data, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly_data)
    writer.SetFileName(str(path))
    return writer.Write()
