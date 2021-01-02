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
    fin = ini + size
    ini -= pad
    fin += pad
    image_size = np.array(label_image.GetSize())
    ini[ini < 0] = 0
    for i in range(3):
        fin[i] = min(fin[i], image_size[i])
    size = fin - ini
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
    image.CopyInformation(reference)


def sitk_and(image_a, image_b):
    """
    Thin wrapper of sitk.And to handle errors more elegantly
    """
    image_a_size = image_a.GetSize()
    image_b_size = image_b.GetSize()
    if image_a_size != image_b_size:
        message = (
            f'Sizes of image_a ({image_a_size})'
            ' and image_b ({image_b_size}) do not match'
        )
        raise ValueError(message)
    image_a = sitk.Cast(image_a, sitk.sitkUInt8)
    image_b = sitk.Cast(image_b, sitk.sitkUInt8)
    return sitk.And(image_a, image_b)


def get_random_voxel(mask, border=False, verbose=False):
    if border:
        image = sitk.BinaryContour(mask)
    else:
        image = mask
    array = sitk.GetArrayViewFromImage(image)
    coords = np.array(np.where(array)).T  # N x 3
    N = len(coords)
    if not N:
        coords_voxel = None
    else:
        random_index = torch.randint(N, (1,)).item()
        coords_voxel = coords[random_index]
        coords_voxel = [int(n) for n in reversed(coords_voxel)]  # NumPy vs ITK
    return coords_voxel


def get_random_voxel_ras(mask):
    voxel = get_random_voxel(mask)
    if voxel is None:
        return None
    center_lps = mask.TransformIndexToPhysicalPoint(tuple(voxel))
    l, p, s = center_lps
    center_ras = -l, -p, s
    return center_ras


def get_cuboid_image(radii_world, reference: sitk.Image, center_ras):
    r, a, s = center_ras
    center_lps = -r, -a, s
    center_voxel = reference.TransformPhysicalPointToIndex(center_lps)
    spacing = np.array(reference.GetSpacing())
    radii_voxel = np.array(radii_world) / spacing
    radii_voxel = radii_voxel.round().astype(np.uint16)
    axes_voxel = 2 * radii_voxel
    cuboid = sitk.Image(*axes_voxel.tolist(), sitk.sitkUInt8) + 1
    result = reference * 0
    destination = (center_voxel - radii_voxel).tolist()
    paste = sitk.PasteImageFilter()
    paste.SetDestinationIndex(destination)
    paste.SetSourceSize(cuboid.GetSize())
    result = paste.Execute(result, cuboid)
    return result


def empty(image):
    return sitk.GetArrayViewFromImage(image).sum() == 0


def erode_bounding_box(image, radius):
    bb = get_bounding_box(image)
    index, size = bb[:3], bb[3:]
    subvolume = get_subvolume(image, bb)
    eroded = sitk.BinaryErode(subvolume, 3 * (radius,))
    result = image * 0
    paste = sitk.PasteImageFilter()
    paste.SetDestinationIndex(index)
    paste.SetSourceSize(size)
    eroded = paste.Execute(result, eroded)
    return eroded
