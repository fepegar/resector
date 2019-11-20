# -*- coding: utf-8 -*-

"""Main module."""

from pathlib import Path
import numpy as np
import SimpleITK as sitk

# pylint: disable=no-member


def resect(
        input_path,
        output_path,
        parcellation_path,
        hemisphere,
        radius=None,
        ):
    if radius is None:
        # From the dataset (although it looks bimodal)
        mean = 10000
        std = 5900
        volume = np.random.normal(mean, std)
        radius = (3/4 * 5900 * np.pi)**(1/3)
    brain = read(input_path)
    gray_matter_mask = get_gray_matter_mask(parcellation_path, hemisphere)
    center = get_voxel_on_border(gray_matter_mask)
    hemisphere_mask = get_resectable_hemisphere_mask(
        parcellation_path, hemisphere)
    write(hemisphere_mask, '/tmp/hemisphere_seg.nii.gz')
    sphere_image = get_sphere_image(hemisphere_mask, center, radius)
    resection_mask = sitk.And(hemisphere_mask, sphere_image)
    resection_mask = sitk.Cast(resection_mask, brain.GetPixelID())
    resected_brain = sitk.MaskNegated(brain, resection_mask)
    write(resected_brain, output_path)


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


def get_resectable_hemisphere_mask(parcellation_path, hemisphere):
    assert hemisphere in ('left', 'right')
    parcellation = read(parcellation_path)
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


def get_gray_matter_mask(parcellation_path, hemisphere):
    """
    There must be a better way of getting GM from GIF
    """
    assert hemisphere in ('left', 'right')
    parcellation = read(parcellation_path)
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
    input_path = Path('~/Dropbox/MRI/t1.nii.gz').expanduser()
    # brain_segmentation_path = Path('~/Dropbox/MRI/t1_brain_seg.nii.gz').expanduser()
    parcellation_path = Path('~/Dropbox/MRI/t1_seg_gif.nii.gz').expanduser()
    for i in range(10):
        output_path = Path(f'/tmp/resected_{i}.nii.gz')
        hemisphere = 'left' if np.random.rand() > 0.5 else 'right'
        resect(
            input_path,
            output_path,
            parcellation_path,
            hemisphere,
        )


if __name__ == "__main__":
    main()
