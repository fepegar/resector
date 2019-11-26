# -*- coding: utf-8 -*-

"""Main module."""

import sys
import shutil
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from subprocess import call, DEVNULL
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from skimage import filters
from noise import pnoise3, snoise3

# pylint: disable=no-member


def resect(
            input_path,
            parcellation_path,
            noise_image_path,
            output_path,
            resection_mask_output_path,
            hemisphere,
            radius=None,
            opening_radius=None,
        ):
    brain = read(input_path)
    parcellation = read(parcellation_path)
    noise_image = read(noise_image_path)

    if radius is None:
        # From the dataset (although it looks bimodal)
        mean = 10000
        std = 5900
        volume = -1
        while volume < 0:
            volume = np.random.normal(mean, std)
        radius = (3 / 4 * volume * np.pi)**(1 / 3)

    # Blend
    if sigmas is None:
        sigmas = np.random.uniform(low=0.5, high=1, size=3)

    gray_matter_mask = get_gray_matter_mask(parcellation, hemisphere)
    resectable_hemisphere_mask = get_resectable_hemisphere_mask(
        parcellation, hemisphere)
    resected_brain, resection_mask, _ = _resect(
        brain,
        gray_matter_mask,
        resectable_hemisphere_mask,
        noise_image,
        radius,
        sigmas,
        opening_radius=opening_radius,
    )
    write(resected_brain, output_path)
    write(resection_mask, resection_mask_output_path)


def _resect(
            brain,
            gray_matter_mask,
            resectable_hemisphere_mask,
            noise_image,
            volume,
            sigmas,
            opening_radius=None,
        ):
    resection_mask, center_ras = get_resection_mask_from_mesh(
        resectable_hemisphere_mask,
        gray_matter_mask,
        volume,
        opening_radius=opening_radius,
    )
    resected_brain = blend(brain, noise_image, resection_mask, sigmas)
    return resected_brain, resection_mask, center_ras


def get_resection_mask_from_mesh(
            resectable_hemisphere_mask,
            gray_matter_mask,
            volume,
            opening_radius=None,
        ):
    """
    TODO: figure out how to do this with VTK and ITK
    """
    opening_radius = 3 if opening_radius is None else opening_radius
    voxel_center = get_random_voxel(gray_matter_mask, border=False)
    center_ras = gray_matter_mask.TransformIndexToPhysicalPoint(
        tuple(voxel_center))
    r, a, s = center_ras
    l, p, s = -r, -a, s
    center_lps = l, p, s
    with NamedTemporaryFile(suffix='.nii') as reference_file:
        with NamedTemporaryFile(suffix='.vtp') as model_file:
            with NamedTemporaryFile(suffix='.nii') as result_file:
                reference_path = reference_file.name
                model_path = model_file.name
                result_path = result_file.name

                # Create noisy sphere
                with warnings.catch_warnings():
                    """
                    To ignore this warning:
                    https://gitlab.kitware.com/vtk/vtk/merge_requests/4847
                    """
                    warnings.simplefilter("ignore")
                    poly_data = get_noisy_sphere_poly_data(
                        center_lps,
                        volume,
                    )
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetFileName(model_path)
                writer.SetInputData(poly_data)
                writer.Write()

                write(resectable_hemisphere_mask, reference_path)  # TODO: use an existing image

                # There must be something already in result_path so that
                # Slicer can load the vtkMRMLVolumeNode? So many ugly hacks :(
                shutil.copy2(reference_path, result_path)

                mesh_to_volume(reference_path, model_path, result_path)
                sphere_mask = read(result_path)

    # Intersection with resectable area
    resection_mask = sitk.And(resectable_hemisphere_mask, sphere_mask)

    # Binary opening to remove leaked CSF from contralateral hemisphere
    if opening_radius is not None:
        resection_mask = sitk.BinaryMorphologicalOpening(
            resection_mask, opening_radius)

    # Use largest connected component only
    resection_mask = get_largest_connected_component(resection_mask)

    return resection_mask, center_ras


def get_slicer_paths(version='4.11'):
    platform = sys.platform
    if platform == 'darwin':  # macOS
        slicer_dir = Path('/Applications/Slicer.app/Contents')
        slicer_bin_path = slicer_dir / 'MacOS' / 'Slicer'
    elif platform == 'linux':
        slicer_dir = Path('~/opt/Slicer/Nightly/Slicer').expanduser()
        slicer_bin_path = slicer_dir / 'Slicer'
    lib_dir = slicer_dir / 'lib' / f'Slicer-{version}'
    modules_dir = lib_dir / 'cli-modules'
    module_name = 'ModelToLabelMap'
    slicer_module_path = modules_dir / module_name

    import os
    environ_dict = dict(os.environ)
    if 'LD_LIBRARY_PATH' not in environ_dict:
        environ_dict['LD_LIBRARY_PATH'] = ''
    environ_dict['LD_LIBRARY_PATH'] += f':{modules_dir}'
    environ_dict['LD_LIBRARY_PATH'] += f':{lib_dir}'
    os.environ.clear()
    os.environ.update(environ_dict)

    return slicer_bin_path, slicer_module_path


def mesh_to_volume(reference_path, model_path, result_path):
    _, slicer_module_path = get_slicer_paths()
    # command = (
    #     slicer_bin_path,
    #     '--no-splash',
    #     '--no-main-window',
    #     '--ignore-slicerrc',
    #     '--launch', slicer_module_path,
    #     reference_path,
    #     model_path,
    #     result_path,
    # )
    command = (
        slicer_module_path,
        reference_path,
        model_path,
        result_path,
    )
    command = [str(a) for a in command]
    call(command, stderr=DEVNULL, stdout=DEVNULL)


def blend(image, noise_image, mask, sigmas):
    image_cast = sitk.Cast(mask, noise_image.GetPixelID())
    mask = sitk.Cast(mask, noise_image.GetPixelID())

    mask = sitk.SmoothingRecursiveGaussian(mask, sigmas)

    alpha = mask
    image_resected = alpha * noise_image + (1 - alpha) * image
    image_resected = sitk.Cast(image_resected, image.GetPixelID())

    return image_resected


def get_largest_connected_component(image):
    cc = sitk.ConnectedComponent(image)
    labeled_cc = sitk.RelabelComponent(cc)
    largest_cc = labeled_cc == 1
    return largest_cc


def make_noise_image(image_path, parcellation_path, output_path, threshold=True, fit=False):
    image = read(image_path)
    parcellation = read(parcellation_path)
    csf_mask = get_csf_mask(image, parcellation)
    image_array = sitk.GetArrayViewFromImage(image)
    csf_mask_array = sitk.GetArrayViewFromImage(csf_mask) > 0  # to bool needed
    csf_values = image_array[csf_mask_array]
    if threshold:  # remove non-CSF voxels
        otsu = filters.threshold_otsu(csf_values)
        csf_values = csf_values[csf_values < otsu]
    if fit:
        from scipy.stats import rayleigh, rice, norm
        distribution = norm
        args = distribution.fit(csf_values)
        random_variable = distribution(*args)
        noise_array = random_variable.rvs(
            size=image_array.shape).astype(np.float32)
    else:  # assume normal distribution
        noise_array = torch.FloatTensor(image_array.shape)
        noise_array.normal_(csf_values.mean(), csf_values.std()).numpy()
    # noise_array = np.random.choice(csf_values, size=image_array.shape)
    noise_image = get_image_from_reference(noise_array, image)
    write(noise_image, output_path)


def get_masked_image(image, mask):
    mask = sitk.Cast(mask, image.GetPixelID())
    masked = sitk.MaskNegated(image, mask)
    return masked


def get_random_voxel(mask, border=False):
    if border:
        image = sitk.BinaryContour(mask)
    else:
        image = mask
    array = sitk.GetArrayViewFromImage(image)
    coords = np.array(np.where(array)).T  # N x 3
    N = len(coords)
    random_index = torch.randint(N, (1,)).item()
    coords_voxel = coords[random_index]
    coords_voxel = [int(n) for n in reversed(coords_voxel)]  # NumPy vs ITK
    return coords_voxel


def get_random_voxel_image(mask, border=False):
    coords_voxel = get_random_voxel(mask, border=border)
    image = sitk.Image(mask)
    image = sitk.Multiply(image, 0)
    image.SetPixel(*coords_voxel, 1)
    return image


def get_sphere_poly_data(center, radius, theta=None, phi=None):
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(center)
    sphere_source.SetRadius(radius)
    if theta is not None:
        sphere_source.SetThetaResolution(theta)
    if phi is not None:
        sphere_source.SetPhiResolution(phi)
    sphere_source.Update()
    return sphere_source.GetOutput()


def get_noisy_sphere_poly_data(
            center,
            volume,
            theta=None,
            phi=None,
            noise_type=None,
            octaves=None,
            smoothness=None,
            noise_amplitude_radius_ratio=None,
        ):
    # TODO: apply trsf to sphere to make it an ellipsoid

    theta = 64 if theta is None else theta
    phi = 64 if phi is None else phi
    noise_type = 'perlin' if noise_type is None else noise_type
    assert noise_type in 'perlin', 'simplex'
    octaves = 4 if octaves is None else octaves  # make random?
    smoothness = 10 if smoothness is None else smoothness  # make random?
    if noise_amplitude_radius_ratio is None:  # make random?
        noise_amplitude_radius_ratio = 0.75

    radius = (3 / 4 * volume / np.pi)**(1 / 3)

    poly_data = get_sphere_poly_data(center, radius, theta=theta, phi=phi)
    wrap_data_object = dsa.WrapDataObject(poly_data)
    points = wrap_data_object.Points
    normals = wrap_data_object.PointData['Normals']

    noise_amplitude = radius * noise_amplitude_radius_ratio
    if noise_type == 'perlin':
        function = pnoise3
    elif noise_type == 'simplex':
        function = snoise3

    points_with_noise = []
    for point, normal in zip(points, normals):
        x, y, z = point / smoothness
        noise = function(x, y, z, octaves=octaves)  # add random offset?
        perturbance = noise_amplitude * noise
        point_with_noise = point + perturbance * normal
        points_with_noise.append(point_with_noise)
    points_with_noise = np.array(points_with_noise)

    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(points_with_noise))
    poly_data.SetPoints(vertices)

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.SetComputePointNormals(True)
    normals_filter.SetComputeCellNormals(True)
    normals_filter.SplittingOff()
    normals_filter.SetInputData(poly_data)
    normals_filter.Update()
    poly_data = normals_filter.GetOutput()

    return poly_data


def get_sphere_mask(voxel_image, radius, approximate=True):
    """
    Approximate distance map is 10 times faster
    """
    assert radius > 0
    if approximate:
        function = sitk.ApproximateSignedDistanceMap
    else:
        function = sitk.SignedDanielssonDistanceMap
    distance = function(voxel_image)
    sphere_mask = distance < radius
    return sphere_mask


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
    progress = tqdm(lines, leave=False)
    for line in progress:
        if pattern.lower() in line.lower():
            label, name = line.split()[:2]
            label = int(label)
            progress.set_description(f'Removing {name}')
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


def get_resection_mask_old(
            resectable_hemisphere_mask,
            gray_matter_mask,
            radius,
            opening_radius=None,
            method='vtk',
        ):
    voxel_image = get_random_voxel_image(gray_matter_mask, border=False)
    sphere_mask = get_sphere_mask(voxel_image, radius)

    # Intersection with resectable area
    resection_mask = sitk.And(resectable_hemisphere_mask, sphere_mask)

    if method != 'vtk':
        # Add some strels on the border
        resection_mask = add_structural_elements(resection_mask)

    # Binary opening to remove leaked CSF from contralateral hemisphere
    if opening_radius is not None:
        resection_mask = sitk.BinaryMorphologicalOpening(
            resection_mask, opening_radius)

    if method != 'vtk':
        # In case the strels appear outside resectable area
        resection_mask = sitk.And(resectable_hemisphere_mask, resection_mask)

    # Use largest connected component only
    resection_mask = get_largest_connected_component(resection_mask)

    return resection_mask


def add_structural_elements_old(mask, min_radius=None, max_radius=None, iterations=3):
    original_mask = mask
    max_radius = 10 if max_radius is None else max_radius
    min_radius = 3 if min_radius is None else min_radius
    kernels = (
        sitk.sitkBall,
        sitk.sitkBox,
        sitk.sitkCross,
    )
    kernels *= iterations
    num_kernels = len(kernels)
    radii_kernels = torch.randint(
        min_radius,
        max_radius + 1,
        size=(num_kernels, 3),
    )
    zipped = list(zip(kernels, radii_kernels))
    for kernel, radii in tqdm(zipped, leave=False):
        voxel_image = get_random_voxel_image(original_mask, border=True)
        artifact = sitk.BinaryDilate(voxel_image, radii.tolist(), kernel)
        mask = sitk.Or(mask, artifact)
    return mask



# if __name__ == "__main__":
#     np.random.seed(42)
#     # input_path = Path('~/Dropbox/MRI/t1.nii.gz').expanduser()
#     # # brain_segmentation_path = Path('~/Dropbox/MRI/t1_brain_seg.nii.gz').expanduser()
#     # parcellation_path = Path('~/Dropbox/MRI/t1_seg_gif.nii.gz').expanduser()
#     # input_path = '/home/fernando/episurg/mri/mni_alejandro/MNI_152_mri.nii.gz'
#     # parcellation_path = '/home/fernando/episurg/mri/mni_alejandro/MNI_152_gif.nii.gz'

#     output_dir = Path('/tmp/resected_perlin')
#     output_dir.mkdir(exist_ok=True)
#     # if output_dir.is_dir():
#     #     import shutil
#     #     shutil.rmtree(output_dir)

#     N = 10

#     input_path = '/tmp/1395_unbiased.nii.gz'
#     parcellation_path = '/home/fernando/episurg_old/subjects/1395/mri/t1_pre/assessors/1395_t1_pre_gif_seg.nii.gz'
#     noise_image_path = Path('/tmp/1395_unbiased_noise.nii.gz')
#     if not noise_image_path.is_file():
#         make_noise_image(input_path, parcellation_path, noise_image_path)
#     for i in trange(N):
#         output_path = output_dir / f'1395_resected_{i}.nii.gz'
#         resection_mask_output_path = output_dir / f'1395_resected_{i}_label.nii.gz'
#         hemisphere = 'left' if np.random.rand() > 0.5 else 'right'
#         opening_radius = 3
#         resect(
#             input_path,
#             parcellation_path,
#             noise_image_path,
#             output_path,
#             resection_mask_output_path,
#             hemisphere,
#             opening_radius=opening_radius,
#         )
