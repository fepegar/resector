import time
import torch
import numpy as np
import SimpleITK as sitk


def get_largest_connected_component(image):
    connected_components = sitk.ConnectedComponent(image)
    labeled_cc = sitk.RelabelComponent(connected_components)
    largest_cc = labeled_cc == 1
    return largest_cc


def get_bounding_box(label_image, pad=0):
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sitk.Cast(label_image, sitk.sitkUInt8))
    bb = np.array(f.GetBoundingBox(1))
    ini, size = bb[:3], bb[3:]
    ini -= pad
    size += 2 * pad
    return ini.tolist() + size.tolist()


def get_subvolume(image, bounding_box):
    i, j, k, si, sj, sk = bounding_box
    sub_image = image[
        i:i + si,
        j:j + sj,
        k:k + sk,
    ]
    return sub_image


def set_metadata(image: sitk.Image, reference: sitk.Image):
    assert image.GetSize() == reference.GetSize()
    image.SetDirection(reference.GetDirection())
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())


def sitk_and(image_a, image_b):
    """
    Thin wrapper of sitk.And to handle errors more elegantly
    """
    image_a_size = image_a.GetSize()
    image_b_size = image_b.GetSize()
    if image_a_size != image_b_size:
        message = (
            f'Sizes of image_a ({image_a_size}) and image_b ({image_b_size}) do not match'
        )
        raise ValueError(message)
    image_a = sitk.Cast(image_a, sitk.sitkUInt8)
    image_b = sitk.Cast(image_b, sitk.sitkUInt8)
    return sitk.And(image_a, image_b)


def get_random_voxel(mask, border=False, verbose=False):
    if verbose:
        start = time.time()
    if border:
        image = sitk.BinaryContour(mask)
    else:
        image = mask
    array = sitk.GetArrayViewFromImage(image)
    coords = np.array(np.where(array)).T  # N x 3
    N = len(coords)
    random_index = torch.randint(N, (1,)).item()
    coords_voxel = coords[random_index]
    coords_voxel = [int(n) for n in reversed(coords_voxel)]  # NumPy vs ITK
    if verbose:
        duration = time.time() - start
        print(f'get_random_voxel: {duration:.1f} seconds')
    return coords_voxel
