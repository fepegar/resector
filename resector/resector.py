import time

from .texture import blend
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
        sigmas_white_matter=None,
        scale_white_matter=None,
        sphere_poly_data=None,
        noise_offset=None,
        simplex_path=None,
        verbose=False,
        ):

    center_ras = get_random_voxel_ras(gray_matter_mask)

    shape = 'ellipsoid'

    if shape is None:
        # Get normal resection
        noisy_poly_data = get_resection_poly_data(
            center_ras,
            radii,
            angles,
            noise_offset=noise_offset,
            sphere_poly_data=sphere_poly_data,
            verbose=verbose,
        )
        if sigmas_white_matter is not None:
            wm_lesion_poly_data = scale_poly_data(
                resection_poly_data,
                scale_white_matter,
            )

            wm_lesion_mask = None

            wm_lesion_image = None

        raw_resection_mask = mesh_to_volume(
            noisy_poly_data,
            resectable_hemisphere_mask,
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

    return resected_image, resection_mask, center_ras
