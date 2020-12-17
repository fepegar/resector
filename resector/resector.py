import SimpleITK as sitk
from .texture import blend, add_wm_lesion, add_clot
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
from .timer import timer


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
        clot_percentiles=(90, 99),
        sphere_poly_data=None,
        noise_offset=None,
        simplex_path=None,
        verbose=False,
        ):
    image = sitk.Cast(image, sitk.sitkFloat32)

    original_image = image
    center_ras = get_random_voxel_ras(gray_matter_mask)

    if shape is None:
        with timer('Noisy mesh', verbose):
            noisy_poly_data = get_resection_poly_data(
                center_ras,
                radii,
                angles,
                noise_offset=noise_offset,
                sphere_poly_data=sphere_poly_data,
                verbose=verbose,
            )

        with timer('mesh to volume', verbose):
            raw_resection_mask = mesh_to_volume(
                noisy_poly_data,
                resectable_hemisphere_mask,
            )

        if wm_lesion:
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
    assert not_empty(resection_mask), 'Masked resection label is empty'

    if shape is None:  # noisy sphere can generate multiple components?
        # Use largest connected component only
        with timer('largest connected component', verbose):
            resection_mask = get_largest_connected_component(resection_mask)

    with timer('blending', verbose):
        resected_image = blend(
            image,
            noise_image,
            resection_mask,
            sigmas,
            simplex_path=simplex_path,
        )

    center_clot_ras = None
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
                sigmas,
                verbose=verbose,
            )

    return resected_image, resection_mask, center_ras, center_clot_ras
