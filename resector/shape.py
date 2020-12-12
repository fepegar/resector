import numpy as np
import SimpleITK as sitk

from .mesh import mesh_to_volume, get_resection_poly_data
from .image import sitk_and, get_random_voxel, get_largest_connected_component


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
        import time
        start = time.time()
    resection_mask = get_largest_connected_component(resection_mask)
    if verbose:
        duration = time.time() - start
        print(f'Largest connected component: {duration:.1f} seconds')

    return resection_mask, np.array(center_ras)
