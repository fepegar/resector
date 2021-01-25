import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from .timer import timer
from .image import (
    get_subvolume,
    get_bounding_box,
    get_random_voxel_ras,
    sitk_and,
    erode_bounding_box,
)
from .mesh import scale_poly_data, mesh_to_volume, get_resection_poly_data
from .io import save_debug


def sample_simplex_noise(
        simplex_path,
        reference,
        size,
        gamma=1,
        # persistence_index=2,
        ):
    # Gamma expansion
    # This is so that bright artifacts do not happen everywhere
    nii = nib.load(simplex_path)
    ci, cj, ck = crop_shape = np.array(size)[::-1]  # sitk to np
    max_shape = np.array(nii.shape) - crop_shape
    # use values near the border, with higher persistence
    # persistence_index /= persistence_index
    mi, mj, mk = max_shape.round().astype(int).tolist()
    i_ini = torch.randint(0, mi, (1,)).item()
    j_ini = torch.randint(0, mj, (1,)).item()
    k_ini = torch.randint(0, mk, (1,)).item()
    index_ini = i_ini, j_ini, k_ini
    index_ini = 0, 0, 0
    i_fin = i_ini + ci
    j_fin = j_ini + cj
    k_fin = k_ini + ck
    sub_simplex_array = nii.dataobj[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
    sub_simplex_array -= sub_simplex_array.min()
    sub_simplex_array /= sub_simplex_array.max()
    if gamma != 1:
        sub_simplex_array **= gamma
    sub_simplex = sitk.GetImageFromArray(sub_simplex_array)
    sub_simplex.SetOrigin(reference.GetOrigin())
    sub_simplex.SetDirection(reference.GetDirection())
    sub_simplex.SetSpacing(reference.GetSpacing())
    return sub_simplex, index_ini


def map(n, start1, stop1, start2, stop2):
    # https://github.com/processing/p5.js/blob/b15ca7c7ac8ba95ccb0123cf74fc163040090e1b/src/math/calculation.js#L450
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2


def add_simplex_noise(
        sub_noise_image,
        full_image,
        simplex_path,
        percentile=99.5,
        ):
    array = sitk.GetArrayViewFromImage(full_image)
    min_simplex_value, max_simplex_value = np.percentile(array, (0.5, 99.9))
    simplex_patch, index_ini = sample_simplex_noise(
        simplex_path,
        sub_noise_image,
        sub_noise_image.GetSize(),
    )  # [0, 1]
    simplex_patch *= (max_simplex_value - min_simplex_value)
    simplex_patch += min_simplex_value
    # return sub_noise_image + simplex_patch
    return simplex_patch


def get_percentile(image, percentile):
    array = sitk.GetArrayViewFromImage(image)
    return np.percentile(array, percentile)


def blend(
        image,
        texture_image,
        mask,
        sigmas=None,
        simplex_path=None,
        pad=10,
        ):
    bounding_box = get_bounding_box(mask, pad=pad)
    sub_image = get_subvolume(image, bounding_box)
    sub_mask = get_subvolume(mask, bounding_box)
    sub_texture_image = get_subvolume(texture_image, bounding_box)

    if simplex_path is not None:
        sub_texture_image = add_simplex_noise(
            sub_texture_image,
            image,
            simplex_path,
        )

    sub_image = sitk.Cast(sub_image, texture_image.GetPixelID())
    sub_mask = sitk.Cast(sub_mask, texture_image.GetPixelID())
    if sigmas is not None:
        sub_mask = sitk.SmoothingRecursiveGaussian(sub_mask, sigmas)
    alpha = sub_mask

    assert alpha.GetSize() == sub_image.GetSize()
    assert alpha.GetSize() == sub_texture_image.GetSize()
    assert alpha.GetPixelID() == sub_image.GetPixelID()
    assert alpha.GetPixelID() == sub_texture_image.GetPixelID()

    sub_image_resected = alpha * sub_texture_image + (1 - alpha) * sub_image
    sub_image_resected = sitk.Cast(sub_image_resected, image.GetPixelID())

    f = sitk.PasteImageFilter()
    f.SetDestinationIndex(bounding_box[:3])
    f.SetSourceSize(bounding_box[3:])
    image_resected = f.Execute(image, sub_image_resected)
    return image_resected


def get_texture_image(image, noise_image, texture):
    if texture == 'dark':
        new_mean = get_percentile(image, 1)
        texture_image = image * 0 + new_mean
    elif texture == 'random':
        percentile = torch.randint(1, 100, (1,)).item()  # i.e., [1, 99]
        new_mean = get_percentile(image, percentile)
        texture_image = image * 0 + new_mean
    elif texture == 'csf':
        texture_image = noise_image
    else:
        raise RuntimeError(F'Texture not recognized: {texture}')
    return texture_image


def clean_outside_resectable(
        image,
        resected_image,
        resectable_hemisphere_mask,
        gray_matter_mask,
        ):
    resectable_no_gm = sitk.Xor(resectable_hemisphere_mask, gray_matter_mask)
    save_debug(resectable_no_gm)
    clean_resected = blend(
        image,
        resected_image,
        resectable_no_gm,
        (1, 1, 1),
    )
    return clean_resected


def get_bright_noise(image, csf_noise, percentiles):
    """
    Create noise image with same std as input noise image and random mean
    within certain percentiles values of input image
    """
    image_array = sitk.GetArrayViewFromImage(image)
    csf_mean = sitk.GetArrayViewFromImage(csf_noise).mean()
    perc_a, perc_b = np.percentile(image_array, percentiles)
    new_mean = torch.FloatTensor(1).uniform_(perc_a, perc_b).item()
    return csf_noise + (new_mean - csf_mean)


def add_wm_lesion(
        image,
        original_image,
        csf_image,
        poly_data,
        scale_factor,
        center_ras,
        resectable_mask,
        gray_matter_mask,
        sigmas,
        pad,
        verbose=False,
        ):
    wm_lesion_poly_data = scale_poly_data(poly_data, scale_factor, center_ras)
    save_debug(wm_lesion_poly_data)
    with timer('white matter mesh to volume', verbose):
        wm_lesion_mask = mesh_to_volume(wm_lesion_poly_data, resectable_mask)
        save_debug(wm_lesion_mask)
    with timer('white matter blend', verbose):
        image_wm = blend(image, csf_image, wm_lesion_mask, sigmas, pad=pad)
        save_debug(image_wm)
    with timer('white matter clean', verbose):
        image_wm_smooth = clean_outside_resectable(
            original_image, image_wm, resectable_mask, gray_matter_mask)
        save_debug(image_wm_smooth)
    return image_wm_smooth


def add_clot(
        original_image,
        resected_image,
        csf_noise_image,
        resection_mask,
        resection_erosion_radius,
        resection_radii,
        clot_size_ratio_range,
        angles,
        noise_offset,
        sphere_poly_data,
        percentiles,
        sigmas,
        verbose=False,
        ):
    with timer(f'erosion with radius {resection_erosion_radius}', verbose):
        eroded_resection_mask = erode_bounding_box(
            resection_mask, resection_erosion_radius)
        save_debug(eroded_resection_mask)

    with timer('random voxel RAS', verbose):
        center_clot_ras = get_random_voxel_ras(eroded_resection_mask)
    if center_clot_ras is None:
        import warnings
        path = '/tmp/res_mask_seg.nii'
        warnings.warn(f'Eroded resection mask is empty. Saving to {path}')
        sitk.WriteImage(resection_mask, path)
        return resected_image, (-1, -1, -1)
    resection_radii = np.array(resection_radii)
    tensor = torch.FloatTensor(3)
    clot_size_ratios = tensor.uniform_(*clot_size_ratio_range).numpy()
    clot_radii = resection_radii / clot_size_ratios

    with timer('clot poly data', verbose):
        clot_poly_data = get_resection_poly_data(
            center_clot_ras,
            clot_radii,
            angles,
            noise_offset=noise_offset * 2,
            sphere_poly_data=sphere_poly_data,
            verbose=verbose,
        )
        save_debug(clot_poly_data)

    with timer('clot mesh to volume', verbose):
        raw_clot_mask = mesh_to_volume(
            clot_poly_data,
            resection_mask,
        )
        save_debug(raw_clot_mask)

    with timer('intersection', verbose):
        clot_mask = sitk_and(raw_clot_mask, eroded_resection_mask)
        save_debug(clot_mask)

    with timer('bright noise', verbose):
        bright_noise_image = get_bright_noise(
            original_image,
            csf_noise_image,
            percentiles,
        )
        save_debug(bright_noise_image)

    with timer('Blending', verbose):
        resected_image_with_clot = blend(
            resected_image,
            bright_noise_image,
            clot_mask,
            sigmas,
        )
        save_debug(resected_image_with_clot)
    return resected_image_with_clot, center_clot_ras
