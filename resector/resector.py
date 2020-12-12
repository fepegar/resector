# -*- coding: utf-8 -*-

"""Main module."""

import time

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from .mesh import get_resection_poly_data, mesh_to_volume
from .parcellation import get_random_voxel


def resect(
        sphere_poly_data,
        image,
        gray_matter_mask,
        resectable_hemisphere_mask,
        noise_image,
        sigmas,
        radii,
        angles,
        noise_offset=None,
        simplex_path=None,
        verbose=False,
        ):
    resection_mask, center_ras = plan_noisy_ellipsoid(
        sphere_poly_data,
        resectable_hemisphere_mask,
        gray_matter_mask,
        radii,
        angles,
        noise_offset=noise_offset,
        verbose=verbose,
    )
    if verbose:
        start = time.time()
    resected_image = blend(
        image,
        noise_image,
        resection_mask,
        sigmas,
        simplex_path=simplex_path,
    )
    if verbose:
        duration = time.time() - start
        print(f'Blending: {duration:.1f} seconds')
    return resected_image, resection_mask, center_ras


def plan_cuboid(
        gray_matter_mask,
        resectable_hemisphere_mask,
        radii_world,
        ):
    cuboid = get_cuboid_image(radii_world, gray_matter_mask)
    return sitk_and(cuboid, resectable_hemisphere_mask)


def plan_ellipsoid(
        gray_matter_mask,
        resectable_hemisphere_mask,
        radii_world,
        angles,
        sphere_poly_data,
        ):
    ellipsoid_poly_data = get_ellipsoid_poly_data(
        radii_world,
        gray_matter_mask,
        angles,
        sphere_poly_data,
    )
    ellipsoid = mesh_to_volume(ellipsoid_poly_data, gray_matter_mask)
    return sitk_and(ellipsoid, resectable_hemisphere_mask)


def get_cuboid_image(radii_world, gray_matter_mask):
    center_voxel = np.array(get_random_voxel(gray_matter_mask))
    spacing = np.array(gray_matter_mask.GetSpacing())
    radii_voxel = np.array(radii_world) / spacing
    radii_voxel = radii_voxel.round().astype(np.uint16)
    axes_voxel = 2 * radii_voxel
    cuboid = sitk.Image(*axes_voxel.tolist(), sitk.sitkUInt8) + 1
    result = gray_matter_mask * 0
    destination = (center_voxel - radii_voxel).tolist()
    paste = sitk.PasteImageFilter()
    paste.SetDestinationIndex(destination)
    paste.SetSourceSize(cuboid.GetSize())
    result = paste.Execute(result, cuboid)
    return result


def get_ellipsoid_poly_data(
        radii_world,
        gray_matter_mask,
        angles,
        sphere_poly_data,
        ):
    from .mesh import transform_poly_data
    center_voxel = np.array(get_random_voxel(gray_matter_mask))
    l, p, s = gray_matter_mask.TransformIndexToPhysicalPoint(center_voxel.tolist())
    center_ras = -l, -p, s
    ellipsoid = transform_poly_data(
        sphere_poly_data,
        center_ras,
        radii_world,
        angles,
    )
    return ellipsoid


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


def set_metadata(image: sitk.Image, reference: sitk.Image):
    assert image.GetSize() == reference.GetSize()
    image.SetDirection(reference.GetDirection())
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())


def plan_noisy_ellipsoid(
        sphere_poly_data,
        resectable_hemisphere_mask,
        gray_matter_mask,
        radii,
        angles,
        noise_offset=None,
        verbose=False,
        ):
    """
    TODO: figure out how to do this with VTK and ITK
    """
    voxel_center = get_random_voxel(
        gray_matter_mask, border=False, verbose=verbose)
    center_lps = gray_matter_mask.TransformIndexToPhysicalPoint(
        tuple(voxel_center))
    l, p, s = center_lps
    center_ras = -l, -p, s

    noisy_poly_data = get_resection_poly_data(
        sphere_poly_data,
        center_ras,
        radii,
        angles,
        noise_offset=noise_offset,
    )

    sphere_mask = mesh_to_volume(noisy_poly_data, resectable_hemisphere_mask)

    # Intersection with resectable area
    resection_mask = sitk_and(resectable_hemisphere_mask, sphere_mask)

    # Use largest connected component only
    if verbose:
        start = time.time()
    resection_mask = get_largest_connected_component(resection_mask)
    if verbose:
        duration = time.time() - start
        print(f'Largest connected component: {duration:.1f} seconds')

    return resection_mask, np.array(center_ras)


def blend(
        image,
        noise_image,
        mask,
        sigmas,
        simplex_path=None,
        pad=10,
        ):
    bounding_box = get_bounding_box(mask, pad=pad)
    sub_image = get_subvolume(image, bounding_box)
    sub_mask = get_subvolume(mask, bounding_box)
    sub_noise_image = get_subvolume(noise_image, bounding_box)

    if simplex_path is not None:
        sub_noise_image = add_simplex_noise(
            sub_noise_image,
            image,
            simplex_path,
        )

    sub_image = sitk.Cast(sub_image, noise_image.GetPixelID())
    sub_mask = sitk.Cast(sub_mask, noise_image.GetPixelID())
    sub_mask = sitk.SmoothingRecursiveGaussian(sub_mask, sigmas)
    alpha = sub_mask
    sub_image_resected = alpha * sub_noise_image + (1 - alpha) * sub_image
    sub_image_resected = sitk.Cast(sub_image_resected, image.GetPixelID())

    f = sitk.PasteImageFilter()
    f.SetDestinationIndex(bounding_box[:3])
    f.SetSourceSize(bounding_box[3:])
    image_resected = f.Execute(image, sub_image_resected)
    return image_resected


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


def add_simplex_noise(
        sub_noise_image,
        full_image,
        simplex_path,
        percentile=95,
        ):
    array = sitk.GetArrayViewFromImage(full_image)
    max_simplex_value = np.percentile(array, percentile)
    simplex_patch, index_ini = sample_simplex_noise(
        simplex_path,
        sub_noise_image,
        sub_noise_image.GetSize(),
    )  # [0, 1]
    simplex_patch *= max_simplex_value
    print(index_ini)
    return sub_noise_image + simplex_patch


def sample_simplex_noise(
        simplex_path,
        reference,
        size,
        gamma=4,
        persistence_index=2,
        ):
    # Gamma expansion
    # This is so that bright artifacts do not happen everywhere
    nii = nib.load(simplex_path)
    ci, cj, ck = crop_shape = np.array(size)[::-1]  # sitk to np
    max_shape = np.array(nii.shape) - crop_shape
    persistence_index /= persistence_index  # use values near the border, with higher persistence
    mi, mj, mk = max_shape.round().astype(int).tolist()
    i_ini = torch.randint(0, mi, (1,)).item()
    j_ini = torch.randint(0, mj, (1,)).item()
    k_ini = torch.randint(0, mk, (1,)).item()
    index_ini = i_ini, j_ini, k_ini
    i_fin = i_ini + ci
    j_fin = j_ini + cj
    k_fin = k_ini + ck
    sub_simplex_array = nii.dataobj[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
    sub_simplex_array **= gamma
    sub_simplex = sitk.GetImageFromArray(sub_simplex_array)
    sub_simplex.SetOrigin(reference.GetOrigin())
    sub_simplex.SetDirection(reference.GetDirection())
    sub_simplex.SetSpacing(reference.GetSpacing())
    sitk.WriteImage(sub_simplex, '/tmp/fpg/sub_simplex.nii')
    return sub_simplex, index_ini


def map(n, start1, stop1, start2, stop2):
    # https://github.com/processing/p5.js/blob/b15ca7c7ac8ba95ccb0123cf74fc163040090e1b/src/math/calculation.js#L450
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2
