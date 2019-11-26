import SimpleITK as sitk


def read(image_path):
    return sitk.ReadImage(str(image_path))


def write(image, image_path):
    sitk.WriteImage(image, str(image_path))


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
