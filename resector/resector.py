import numpy as np
import SimpleITK as sitk
from .texture import blend, add_wm_lesion, add_clot, get_texture_image
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
    empty,
)
from .timer import timer
from .io import save_debug


def resect(
        image,
        gray_matter_mask,
        resectable_hemisphere_mask,
        sigmas,
        radii,
        noise_image=None,
        shape='noisy',
        texture='csf',
        angles=None,
        sigma_white_matter=10,
        scale_white_matter=3,
        wm_lesion=False,
        clot=False,
        clot_erosion_radius=2,
        clot_size_ratio=(1,1),  ######################(3, 8),
        clot_percentiles=(90, 99),
        sphere_poly_data=None,
        noise_offset=None,
        center_ras=None,
        simplex_path=None,
        verbose=False,
        ):
    image = sitk.Cast(image, sitk.sitkFloat32)

    if texture == 'csf' and noise_image is None:
        raise RuntimeError('CSF image is needed if texture is "csf"')

    original_image = image
    if not center_ras or center_ras is None:  # could be an empty tuple
        center_ras = get_random_voxel_ras(gray_matter_mask)

    if shape == 'noisy':
        with timer('Noisy mesh', verbose):
            noisy_poly_data = get_resection_poly_data(
                center_ras,
                radii,
                angles,
                noise_offset=noise_offset,
                sphere_poly_data=sphere_poly_data,
                verbose=verbose,
            )
            save_debug(noisy_poly_data)

        with timer('mesh to volume', verbose):
            raw_resection_mask = mesh_to_volume(
                noisy_poly_data,
                resectable_hemisphere_mask,
            )
            save_debug(raw_resection_mask)

        if wm_lesion:
            assert noise_image is not None
            with timer('white matter lesion', verbose):
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
        with timer('cuboid image', verbose):
            raw_resection_mask = get_cuboid_image(
                radii,
                gray_matter_mask,
                center_ras,
            )
    elif shape == 'ellipsoid':
        with timer('ellipsoid', verbose):
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

    with timer('intersection with resectable', verbose):
        resection_mask = sitk_and(
            raw_resection_mask, resectable_hemisphere_mask)
        save_debug(resection_mask)
    if empty(resection_mask):
        raise RuntimeError('Resection mask empty')

    if shape == 'noisy':  # noisy sphere can generate multiple components?
        # Use largest connected component only
        with timer('largest connected component', verbose):
            resection_mask = get_largest_connected_component(resection_mask)

    with timer('blending', verbose):
        texture_image = get_texture_image(image, noise_image, texture)
        assert texture_image is not None
        resected_image = blend(
            image,
            texture_image,
            resection_mask,
            sigmas,
            simplex_path=simplex_path,
        )
        save_debug(resected_image)

    center_clot_ras = np.array(3 * (np.nan,))
    if clot:
        with timer('clot', verbose):
            resected_image, center_clot_ras = add_clot(
                original_image,
                resected_image,
                noise_image,
                resection_mask,
                clot_erosion_radius,
                radii,
                clot_size_ratio,
                angles,
                noise_offset,
                sphere_poly_data,
                clot_percentiles,
                (0.5, 0.5, 0.5),
                verbose=verbose,
            )

    return resected_image, resection_mask, center_ras, center_clot_ras
