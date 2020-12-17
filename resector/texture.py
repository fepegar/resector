import time

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from .mesh import scale_poly_data, mesh_to_volume
from .image import get_subvolume, get_bounding_box


def sample_simplex_noise(
        simplex_path,
        reference,
        size,
        gamma=4,
        # persistence_index=2,
        ):
    # Gamma expansion
    # This is so that bright artifacts do not happen everywhere
    nii = nib.load(simplex_path)
    ci, cj, ck = crop_shape = np.array(size)[::-1]  # sitk to np
    max_shape = np.array(nii.shape) - crop_shape
    # persistence_index /= persistence_index  # use values near the border, with higher persistence
    mi, mj, mk = max_shape.round().astype(int).tolist()
    i_ini = torch.randint(0, mi, (1,)).item()
    j_ini = torch.randint(0, mj, (1,)).item()
    k_ini = torch.randint(0, mk, (1,)).item()
    index_ini = i_ini, j_ini, k_ini
    i_fin = i_ini + ci
    j_fin = j_ini + cj
    k_fin = k_ini + ck
    sub_simplex_array = nii.dataobj[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
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
        percentile=95,
        ):
    array = sitk.GetArrayViewFromImage(full_image)
    max_simplex_value = np.percentile(array, percentile)
    simplex_patch, index_ini = sample_simplex_noise(
        simplex_path,
        sub_noise_image,
        sub_noise_image.GetSize(),
    )  # [0, 1]
    simplex_patch *= max_simplex_value
    return sub_noise_image + simplex_patch


def blend(
        image,
        noise_image,
        mask,
        sigmas,
        simplex_path=None,
        pad=10,
        ):
    assert image.GetSize() == noise_image.GetSize()
    assert image.GetSize() == mask.GetSize()
    bounding_box = get_bounding_box(mask, pad=pad)
    sub_image = get_subvolume(image, bounding_box)
    sub_mask = get_subvolume(mask, bounding_box)
    sub_noise_image = get_subvolume(noise_image, bounding_box)

    if simplex_path is not None:
        sub_noise_image = add_simplex_noise(
            sub_noise_image,
            image,
            simplex_path,
        )

    sub_image = sitk.Cast(sub_image, noise_image.GetPixelID())
    sub_mask = sitk.Cast(sub_mask, noise_image.GetPixelID())
    sub_mask = sitk.SmoothingRecursiveGaussian(sub_mask, sigmas)
    alpha = sub_mask

    assert alpha.GetSize() == sub_image.GetSize()
    assert alpha.GetSize() == sub_noise_image.GetSize()
    assert alpha.GetPixelID() == sub_image.GetPixelID()
    assert alpha.GetPixelID() == sub_noise_image.GetPixelID()

    sub_image_resected = alpha * sub_noise_image + (1 - alpha) * sub_image
    sub_image_resected = sitk.Cast(sub_image_resected, image.GetPixelID())

    f = sitk.PasteImageFilter()
    f.SetDestinationIndex(bounding_box[:3])
    f.SetSourceSize(bounding_box[3:])
    image_resected = f.Execute(image, sub_image_resected)
    return image_resected


def clean_outside_resectable(
        image,
        resected_image,
        resectable_hemisphere_mask,
        gray_matter_mask,
        ):
    mask = sitk.Xor(resectable_hemisphere_mask, gray_matter_mask)
    clean_resected = blend(
        image,
        resected_image,
        mask,
        (1, 1, 1),
    )
    return clean_resected

def get_bright_noise(image, csf_noise, percentiles):
    """
    Create noise image with same std as input noise image and random mean within
    the foregound values of input image
    """
    image_array = sitk.GetArrayViewFromImage(image)
    csf_mean = sitk.GetArrayViewFromImage(csf_noise).mean()
    min_percentile, max_percentile = 90, 99
    perc_a = np.percentile(image_array, min_percentile)
    perc_b = np.percentile(image_array, max_percentile)
    new_mean = torch.FloatTensor(1).uniform_(perc_a, perc_b).item()
    return csf_noise + (new_mean - csf_mean)


def add_wm_lesion(
        image,
        original_image,
        csf_noise_image,
        poly_data,
        scale_factor,
        center_ras,
        resectable_mask,
        gray_matter_mask,
        sigmas,
        pad,
        verbose=False,
        ):
    if verbose:
        start = time.time()

    wm_lesion_poly_data = scale_poly_data(
        poly_data,
        scale_factor,
        center_ras,
    )

    wm_lesion_mask = mesh_to_volume(
        wm_lesion_poly_data,
        resectable_mask,
    )

    image = blend(
        image,
        csf_noise_image,
        wm_lesion_mask,
        sigmas,
        pad=pad,
    )

    image = clean_outside_resectable(
        original_image,
        image,
        resectable_mask,
        gray_matter_mask,
    )
    if verbose:
        duration = time.time() - start
        print(f'White matter lesion: {duration:.1f} seconds')

    return image
