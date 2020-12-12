import time

from .texture import blend
from .shape import plan_noisy_ellipsoid


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
