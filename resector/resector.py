# -*- coding: utf-8 -*-

"""Main module."""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage import filters
from scipy.stats import rayleigh, rice, norm

# pylint: disable=no-member


def resect(
        input_path,
        parcellation_path,
        noise_image_path,
        output_path,
        hemisphere,
        radius=None,
        opening_radius=None,
        ):
    if radius is None:
        # From the dataset (although it looks bimodal)
        mean = 10000
        std = 5900
        volume = np.random.normal(mean, std)
        radius = (3/4 * volume * np.pi)**(1/3)
    brain = read(input_path)
    parcellation = read(parcellation_path)
    gray_matter_mask = get_gray_matter_mask(parcellation, hemisphere)
    center = get_voxel_on_border(gray_matter_mask)
    hemisphere_mask = get_resectable_hemisphere_mask(
        parcellation, hemisphere)
    # write(hemisphere_mask, '/tmp/hemisphere_seg.nii.gz')
    sphere_image = get_sphere_image(hemisphere_mask, center, radius)
    noise_image = read(noise_image_path)
    resection_mask = sitk.And(hemisphere_mask, sphere_image)
    if opening_radius is not None:
        resection_mask = sitk.BinaryMorphologicalOpening(
            resection_mask, opening_radius)
    resected_brain = get_masked_image(brain, resection_mask)
    write(resected_brain, output_path)


def make_noise_image(image_path, parcellation_path, output_path, threshold=True):
    image = read(image_path)
    parcellation = read(parcellation_path)
    csf_mask = get_csf_mask(image, parcellation)
    write(csf_mask, '/tmp/csf_seg.nii.gz')
    image_array = sitk.GetArrayViewFromImage(image)
    csf_mask_array = sitk.GetArrayViewFromImage(csf_mask) > 0  # to bool needed
    csf_values = image_array[csf_mask_array]
    if threshold:  # remove non-CSF voxels
        otsu = filters.threshold_otsu(csf_values)
        csf_values = csf_values[csf_values < otsu]
    np.save('/tmp/csf_values.npy', csf_values)
    distribution = norm  ##
    args = distribution.fit(csf_values)
    random_variable = distribution(*args)
    noise_array = random_variable.rvs(size=image_array.shape)
    # noise_array = np.random.choice(csf_values, size=image_array.shape)
    noise_image = get_image_from_reference(noise_array, image)
    write(noise_image, output_path)


def get_masked_image(image, mask):
    mask = sitk.Cast(mask, image.GetPixelID())
    masked = sitk.MaskNegated(image, mask)
    return masked


def get_voxel_on_border(mask):
    border = sitk.BinaryContour(mask)
    border_array = sitk.GetArrayViewFromImage(border)
    coords = np.array(np.where(border_array)).T  # N x 3
    N = len(coords)
    random_index = np.random.randint(N)
    border_voxel = coords[random_index]
    return border_voxel


def get_sphere_image(reference, center, radius):
    image = sitk.Image(reference)
    image = sitk.Multiply(image, 0)
    center = reversed(center.tolist())  # NumPy vs ITK
    image.SetPixel(*center, 1)
    distance = sitk.SignedDanielssonDistanceMap(image)
    sphere_image = distance < radius
    return sphere_image


def get_resectable_hemisphere_mask(parcellation, hemisphere):
    assert hemisphere in ('left', 'right')
    parcellation = sitk.Cast(parcellation, sitk.sitkUInt8)
    array = sitk.GetArrayFromImage(parcellation)
    hemisphere_to_remove = 'left' if hemisphere == 'right' else 'right'
    array[array == 2] = 0  # remove external labels
    array[array == 3] = 0  # remove external labels
    remove_hemisphere(array, hemisphere_to_remove)
    remove_brainstem_and_cerebellum(array)
    # remove_ventricles(array)
    remove_pattern(array, 'Ventral-DC')  # superior part of the brainstem
    mask = get_image_from_reference(array, parcellation) > 0
    return mask


def get_gray_matter_mask(parcellation, hemisphere):
    """
    There must be a better way of getting GM from GIF
    """
    assert hemisphere in ('left', 'right')
    parcellation = sitk.Cast(parcellation, sitk.sitkUInt8)
    array = sitk.GetArrayFromImage(parcellation)
    hemisphere_to_remove = 'left' if hemisphere == 'right' else 'right'
    array[array < 5] = 0  # remove CSF
    remove_hemisphere(array, hemisphere_to_remove)
    remove_brainstem_and_cerebellum(array)
    remove_pattern(array, 'Callosum')
    remove_ventricles(array)
    remove_pattern(array, 'white')
    remove_pattern(array, 'caudate')
    remove_pattern(array, 'putamen')
    remove_pattern(array, 'thalamus')
    remove_pattern(array, 'Ventral-DC')
    mask = get_image_from_reference(array, parcellation) > 0
    return mask


def get_csf_mask(image, parcellation, erode_radius=1):
    parcellation_array = sitk.GetArrayFromImage(parcellation)
    parcellation_array[parcellation_array == 1] = 0  # should I remove this?
    parcellation_array[parcellation_array == 2] = 0
    parcellation_array[parcellation_array == 3] = 0
    parcellation_array[parcellation_array == 4] = 0
    lines = get_color_table()
    for line in lines:
        line = line.lower()
        if 'periventricular' in line: continue
        if not 'ventric' in line.lower():
            label, name = line.split()[:2]
            label = int(label)
            print('Removing', name)
            parcellation_array[parcellation_array == label] = 0
    csf_mask_array = parcellation_array > 0
    csf_mask = get_image_from_reference(csf_mask_array, image)
    if erode_radius is not None:
        csf_mask = sitk.BinaryErode(csf_mask, erode_radius)
    return csf_mask


def remove_hemisphere(array, hemisphere):
    remove_pattern(array, hemisphere)


def remove_brainstem_and_cerebellum(array):
    remove_pattern(array, 'cerebell')
    remove_pattern(array, 'brain-stem')
    remove_pattern(array, 'pons')


def remove_ventricles(array):
    remove_pattern(array, '-ventric')


def remove_pattern(array, pattern):
    lines = get_color_table()
    for line in lines:
        if pattern.lower() in line.lower():
            label, name = line.split()[:2]
            label = int(label)
            print('Removing', name)
            array[array == label] = 0


def get_color_table():
    labels_path = Path(__file__).parent.parent / 'BrainAnatomyLabelsV3_0.txt'
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


def read(image_path):
    return sitk.ReadImage(str(image_path))


def write(image, image_path):
    sitk.WriteImage(image, str(image_path))


def main():
    np.random.seed(42)
    # input_path = Path('~/Dropbox/MRI/t1.nii.gz').expanduser()
    # # brain_segmentation_path = Path('~/Dropbox/MRI/t1_brain_seg.nii.gz').expanduser()
    # parcellation_path = Path('~/Dropbox/MRI/t1_seg_gif.nii.gz').expanduser()
    # input_path = '/home/fernando/episurg/mri/mni_alejandro/MNI_152_mri.nii.gz'
    # parcellation_path = '/home/fernando/episurg/mri/mni_alejandro/MNI_152_gif.nii.gz'

    input_path = '/tmp/1395_unbiased.nii.gz'
    parcellation_path = '/home/fernando/episurg_old/subjects/1395/mri/t1_pre/assessors/1395_t1_pre_gif_seg.nii.gz'
    noise_image_path = Path('/tmp/1395_unbiased_noise.nii.gz')
    if True:  # not noise_image_path.is_file():
        make_noise_image(input_path, parcellation_path, noise_image_path)
    for i in range(1):
        output_path = Path(f'/tmp/resected_{i}.nii.gz')
        hemisphere = 'left' if np.random.rand() > 0.5 else 'right'
        opening_radius = 3
        resect(
            input_path,
            parcellation_path,
            noise_image_path,
            output_path,
            hemisphere,
            opening_radius=opening_radius,
        )


if __name__ == "__main__":
    main()
