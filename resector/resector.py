# -*- coding: utf-8 -*-

"""Main module."""

import time
import warnings
from math import tau
from tempfile import NamedTemporaryFile

import numpy as np
import SimpleITK as sitk

from .io import read, write
from .mesh import get_ellipsoid_poly_data, add_noise_to_poly_data, mesh_to_volume
from .parcellation import (
    get_gray_matter_mask,
    get_resectable_hemisphere_mask,
    get_random_voxel,
)


def resect(
        input_path,
        parcellation_path,
        noise_image_path,
        output_path,
        resection_mask_output_path,
        hemisphere,
        radius,
        sigmas=None,
        opening_radius=None,
        ):
    """
    TODO: fix this?
    """
    brain = read(input_path)
    parcellation = read(parcellation_path)
    noise_image = read(noise_image_path)

    # Blend
    if sigmas is None:
        sigmas = np.random.uniform(low=0.2, high=1, size=3)

    gray_matter_mask = get_gray_matter_mask(parcellation, hemisphere)
    resectable_hemisphere_mask = get_resectable_hemisphere_mask(
        parcellation, hemisphere, opening_radius=opening_radius)

    resected_brain, resection_mask, _ = _resect(
        brain,
        gray_matter_mask,
        resectable_hemisphere_mask,
        noise_image,
        radius,
        sigmas,
    )
    write(resected_brain, output_path)
    write(resection_mask, resection_mask_output_path)


def _resect(
        brain,
        gray_matter_mask,
        resectable_hemisphere_mask,
        noise_image,
        volume,
        sigmas,
        radii_ratio,
        angles,
        verbose=False,
        ):
    resection_mask, center_ras = get_resection_mask_from_mesh(
        resectable_hemisphere_mask,
        gray_matter_mask,
        volume,
        radii_ratio,
        angles,
        verbose=verbose,
    )
    if verbose:
        start = time.time()
    resected_brain = blend(brain, noise_image, resection_mask, sigmas)
    if verbose:
        duration = time.time() - start
        print(f'Blending: {duration:.1f} seconds')
    return resected_brain, resection_mask, center_ras


def get_resection_mask_from_mesh(
        resectable_hemisphere_mask,
        gray_matter_mask,
        volume,
        radii_ratio,
        angles,
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
        with NamedTemporaryFile(suffix='.nii') as result_file:
            reference_path = reference_file.name
            result_path = result_file.name

            # As the center of the sphere lies at the border of the brain,
            # volume should be the volume of a hemisphere. We ignore this to
            # take into account gray matter that is in deep sulci and to
            # create smaller resections, i.e. harder cases
            radius = (3 / 2 * volume / tau)**(1 / 3)

            poly_data = get_ellipsoid_poly_data(
                center_ras,
                radius,
                radii_ratio,
                angles,
            )
            with warnings.catch_warnings():
                # To ignore this warning:
                # https://gitlab.kitware.com/vtk/vtk/merge_requests/4847
                warnings.simplefilter("ignore")
                poly_data = add_noise_to_poly_data(poly_data, radius, verbose=verbose)

            write(resectable_hemisphere_mask, reference_path)  # TODO: use an existing image
            sphere_mask = mesh_to_volume(poly_data, reference_path, result_path)

    # Intersection with resectable area
    resection_mask = sitk.And(resectable_hemisphere_mask, sphere_mask)

    # Use largest connected component only
    if verbose:
        start = time.time()
    resection_mask = get_largest_connected_component(resection_mask)
    if verbose:
        duration = time.time() - start
        print(f'Largest connected component: {duration:.1f} seconds')

    return resection_mask, center_ras


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
