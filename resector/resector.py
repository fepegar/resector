import time

import torch
import numpy as np
import SimpleITK as sitk
from .texture import blend, add_wm_lesion, get_bright_noise
from .mesh import (
    get_resection_poly_data,
    get_ellipsoid_poly_data,
    mesh_to_volume,
)
from .image import (
    get_largest_connected_component,
    sitk_and,
    get_random_voxel_ras,
    get_cuboid_image,
    not_empty,
)


def resect(
        image,
        gray_matter_mask,
        resectable_hemisphere_mask,
        noise_image,
        sigmas,
        radii,
        shape=None,
        angles=None,
        sigma_white_matter=10,
        scale_white_matter=3,
        wm_lesion=False,
        clot=False,
        clot_erosion_radius=1,
        clot_size_ratio=(1.5, 3),
        sphere_poly_data=None,
        noise_offset=None,
        simplex_path=None,
        verbose=False,
        ):
    image = sitk.Cast(image, sitk.sitkFloat32)

    original_image = image
    center_ras = get_random_voxel_ras(gray_matter_mask)

    if shape is None:
        # Get normal resection
        if verbose:
            start = time.time()
        noisy_poly_data = get_resection_poly_data(
            center_ras,
            radii,
            angles,
            noise_offset=noise_offset,
            sphere_poly_data=sphere_poly_data,
            verbose=verbose,
        )
        if verbose:
            duration = time.time() - start
            print(f'Noisy mesh: {duration:.1f} seconds')

        raw_resection_mask = mesh_to_volume(
            noisy_poly_data,
            resectable_hemisphere_mask,
        )

        if wm_lesion:
            image = add_wm_lesion(
                image,
                original_image,
                noise_image,
                noisy_poly_data,
                scale_white_matter,
                center_ras,
                resectable_hemisphere_mask,
                gray_matter_mask,
                3 * (sigma_white_matter,),
                pad=20,
                verbose=verbose,
            )

    elif shape == 'cuboid':
        raw_resection_mask = get_cuboid_image(
            radii,
            gray_matter_mask,
            center_ras,
        )
    elif shape == 'ellipsoid':
        poly_data = get_ellipsoid_poly_data(
            radii,
            center_ras,
            angles,
            sphere_poly_data=sphere_poly_data,
        )
        raw_resection_mask = mesh_to_volume(
            poly_data,
            resectable_hemisphere_mask,
        )

    resection_mask = sitk_and(raw_resection_mask, resectable_hemisphere_mask)
    assert not_empty(resection_mask), 'Masked resection label is empty'

    if shape is None:  # noisy sphere can generate multiple components?
        # Use largest connected component only
        if verbose:
            start = time.time()
        resection_mask = get_largest_connected_component(resection_mask)
        if verbose:
            duration = time.time() - start
            print(f'Largest connected component: {duration:.1f} seconds')

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

    center_clot_ras = None
    if clot:
        if verbose:
            start = time.time()
        eroded_resection_mask = sitk.BinaryErode(
            resection_mask,
            3 * [clot_erosion_radius],
        )
        center_clot_ras = get_random_voxel_ras(eroded_resection_mask)
        radii = np.array(radii)
        clot_size_ratios = torch.FloatTensor(3).uniform_(*clot_size_ratio).numpy()
        clot_radii = radii / clot_size_ratios
        clot_poly_data = get_resection_poly_data(
            center_clot_ras,
            clot_radii,
            angles,
            noise_offset=noise_offset * 2,
            sphere_poly_data=sphere_poly_data,
            verbose=verbose,
        )
        raw_clot_mask = mesh_to_volume(
            clot_poly_data,
            resectable_hemisphere_mask,
        )

        clot_mask = sitk_and(raw_clot_mask, eroded_resection_mask)
        bright_noise_image = get_bright_noise(
            original_image,
            noise_image,
            (90, 99),
        )
        resected_image = blend(
            resected_image,
            bright_noise_image,
            clot_mask,
            sigmas,
            simplex_path=simplex_path,
        )
        if verbose:
            duration = time.time() - start
            print(f'Clot: {duration:.1f} seconds')

    return resected_image, resection_mask, center_ras, center_clot_ras
