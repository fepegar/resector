import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
import SimpleITK as sitk
from skimage import filters

from .io import write, nib_to_sitk
from .image import get_largest_connected_component


def get_resectable_hemisphere_mask(
        parcellation_path,
        hemisphere,
        opening_radius=3,
        median_radius=4,
        ):
    assert hemisphere in ('left', 'right')
    parcellation_nii = nib.load(str(parcellation_path))
    array = np.asanyarray(parcellation_nii.dataobj).astype(np.uint8)
    hemisphere_to_remove = 'left' if hemisphere == 'right' else 'right'
    array[array == 1] = 0  # remove external labels
    array[array == 2] = 0  # remove external labels
    array[array == 3] = 0  # remove external labels
    remove_hemisphere(array, hemisphere_to_remove)
    remove_brainstem_and_cerebellum(array)
    mask = nib_to_sitk(array, parcellation_nii.affine) > 0
    mask = sitk.BinaryErode(mask, 3 * [opening_radius])
    mask = get_largest_connected_component(mask)
    mask = sitk.BinaryDilate(mask, 3 * [opening_radius])
    mask = sitk.Median(mask, 3 * (median_radius,))
    return mask


def get_gray_matter_mask(parcellation_path, hemisphere):
    """
    There must be a better way of getting GM from GIF
    """
    assert hemisphere in ('left', 'right')
    parcellation_nii = nib.load(str(parcellation_path))
    array = np.asanyarray(parcellation_nii.dataobj).astype(np.uint8)
    hemisphere_to_remove = 'left' if hemisphere == 'right' else 'right'
    array[array < 5] = 0  # remove CSF
    remove_hemisphere(array, hemisphere_to_remove)
    remove_brainstem_and_cerebellum(array)
    remove_pattern(array, 'Callosum')
    remove_ventricles(array)
    remove_pattern(array, 'white')
    remove_pattern(array, 'caudate')
    remove_pattern(array, 'putamen')
    remove_pattern(array, 'pallidum')
    remove_pattern(array, 'thalamus')
    mask = nib_to_sitk(array, parcellation_nii.affine) > 0
    return mask


def get_white_matter_mask(parcellation_path, hemisphere):
    assert hemisphere in ('left', 'right')
    parcellation_nii = nib.load(str(parcellation_path))
    parcellation = np.asanyarray(parcellation_nii.dataobj).astype(np.uint8)
    array = np.zeros_like(parcellation)
    lines = get_color_table()
    progress = tqdm(lines, leave=False)
    for line in progress:
        line = line.lower()
        if 'white' in line and hemisphere in line:
            label, name = line.split()[:2]
            label = int(label)
            array[parcellation == label] = 1
    mask = nib_to_sitk(array, parcellation_nii.affine) > 0
    return mask


def get_csf_mask(parcellation_path, erode_radius=1) -> sitk.Image:
    parcellation_nii = nib.load(str(parcellation_path))
    parcellation_array = np.asanyarray(parcellation_nii.dataobj)
    parcellation_array = parcellation_array.astype(np.uint8)
    parcellation_array[parcellation_array == 1] = 0  # should I remove this?
    parcellation_array[parcellation_array == 2] = 0
    parcellation_array[parcellation_array == 3] = 0
    parcellation_array[parcellation_array == 4] = 0
    lines = get_color_table()
    progress = tqdm(lines, leave=False)
    for line in progress:
        line = line.lower()
        if 'periventricular' in line:
            continue
        if not 'ventric' in line.lower():
            label, name = line.split()[:2]
            label = int(label)
            progress.set_description(f'Removing {name}')
            parcellation_array[parcellation_array == label] = 0
    csf_mask_array = (parcellation_array > 0).astype(np.uint8)
    csf_mask = nib_to_sitk(csf_mask_array, parcellation_nii.affine)
    if erode_radius is not None:
        csf_mask = sitk.BinaryErode(csf_mask, 3 * [erode_radius])
    return csf_mask


def remove_hemisphere(array, hemisphere):
    remove_pattern(array, hemisphere)


def remove_brainstem_and_cerebellum(array):
    remove_pattern(array, 'cerebell')
    remove_pattern(array, 'brain-stem')
    remove_pattern(array, 'pons')
    remove_pattern(array, 'Ventral-DC')


def remove_ventricles(array):
    remove_pattern(array, '-ventric')


def remove_pattern(array, pattern):
    lines = get_color_table()
    progress = tqdm(lines, leave=False)
    for line in progress:
        if pattern.lower() in line.lower():
            label, name = line.split()[:2]
            label = int(label)
            progress.set_description(f'Removing {name}')
            array[array == label] = 0


def get_color_table():
    labels_path = Path(__file__).parent / 'BrainAnatomyLabelsV3_0.txt'
    lines = labels_path.read_text().splitlines()
    return lines


def get_image_from_reference(array, reference):
    if array.dtype == np.bool:
        array = array.astype(np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    image.SetSpacing(reference.GetSpacing())
    return image


def get_mean_std_threshold(csf_values, threshold, std_factor):
    if threshold:  # remove non-CSF voxels
        otsu = filters.threshold_otsu(csf_values)
        csf_values = csf_values[csf_values < otsu]
    # assume normal distribution
    mean = csf_values.mean()
    std = csf_values.std() * std_factor  # this is because many bright voxels are still included in CSF mask
    return mean, std


def get_mean_std_histogram(csf_values, num_bins):
    count, levels = np.histogram(csf_values, 32)
    max_count_idx = np.argmax(count)
    level_1 = levels[max_count_idx]
    level_2 = levels[max_count_idx + 1]
    mean = np.mean((level_1, level_2))
    std = csf_values[csf_values < mean].std()
    return mean, std


def make_noise_image(
        image_path,
        parcellation_path,
        output_path,
        ):
    image_nii = nib.load(str(image_path))
    csf_mask = get_csf_mask(parcellation_path)
    image_array = np.asanyarray(image_nii.dataobj)
    csf_mask_array = sitk.GetArrayViewFromImage(csf_mask) > 0  # to bool needed
    csf_mask_array = csf_mask_array.transpose(2, 1, 0)  # sitk to np
    csf_values = image_array[csf_mask_array]
    mean, std = get_mean_std_histogram(csf_values, 32)
    noise_tensor = torch.FloatTensor(*image_array.shape)
    noise_tensor = noise_tensor.normal_(mean, std)
    noise_image = nib_to_sitk(noise_tensor.numpy(), image_nii.affine)
    write(noise_image, output_path)
