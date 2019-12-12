# -*- coding: utf-8 -*-

"""Main module."""

import time
import warnings
from tempfile import NamedTemporaryFile

import numpy as np
import SimpleITK as sitk

from .io import write
from .mesh import get_resection_poly_data, mesh_to_volume
from .parcellation import (
    get_gray_matter_mask,
    get_resectable_hemisphere_mask,
    get_random_voxel,
)


def resect(
        sphere_poly_data,
        brain,
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
    resected_brain = blend(brain, noise_image, resection_mask, sigmas)
    if verbose:
        duration = time.time() - start
        print(f'Blending: {duration:.1f} seconds')
    return resected_brain, resection_mask, center_ras


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
    with NamedTemporaryFile(suffix='.nii') as reference_file:
        reference_path = reference_file.name

        noisy_poly_data = get_resection_poly_data(
            sphere_poly_data,
            center_ras,
            radii,
            angles,
            noise_offset=noise_offset,
        )

        # Use image stencil to convert mesh to image
        write(resectable_hemisphere_mask, reference_path)  # TODO: use an existing image
        sphere_mask = mesh_to_volume(noisy_poly_data, reference_path)

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
