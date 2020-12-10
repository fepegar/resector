# -*- coding: utf-8 -*-

"""Main module."""

import time

import numpy as np
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
        verbose=False,
        ):
    resection_mask, center_ras = get_resection_mask_from_mesh(
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
    resected_image = blend(image, noise_image, resection_mask, sigmas)
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
    from .io import write_poly_data
    from .mesh import flipxy
    write_poly_data(flipxy(ellipsoid_poly_data), '/tmp/ellipsoid.vtp')
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


def get_resection_mask_from_mesh(
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


def blend(image, noise_image, mask, sigmas):
    image = sitk.Cast(image, noise_image.GetPixelID())
    mask = sitk.Cast(mask, noise_image.GetPixelID())
    mask = sitk.SmoothingRecursiveGaussian(mask, sigmas)
    alpha = mask
    image_resected = alpha * noise_image + (1 - alpha) * image
    image_resected = sitk.Cast(image_resected, image.GetPixelID())
    return image_resected


def get_largest_connected_component(image):
    connected_components = sitk.ConnectedComponent(image)
    labeled_cc = sitk.RelabelComponent(connected_components)
    largest_cc = labeled_cc == 1
    return largest_cc
