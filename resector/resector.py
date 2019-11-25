# -*- coding: utf-8 -*-

"""Main module."""

from pathlib import Path
from subprocess import call
import numpy as np
from tqdm import tqdm, trange
import SimpleITK as sitk
from skimage import filters
from scipy.stats import rayleigh, rice, norm

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
    if radius is None:
        # From the dataset (although it looks bimodal)
        mean = 10000
        std = 5900
        volume = -1
        while volume < 0:
            volume = np.random.normal(mean, std)
        radius = (3/4 * volume * np.pi)**(1/3)
    brain = read(input_path)
    parcellation = read(parcellation_path)
    gray_matter_mask = get_gray_matter_mask(parcellation, hemisphere)
    resectable_hemisphere_mask = get_resectable_hemisphere_mask(parcellation, hemisphere)

    resection_mask = get_resection_mask(
        resectable_hemisphere_mask,
        gray_matter_mask,
        radius,
        opening_radius=opening_radius,
    )

    # Blend
    sigmas = np.random.uniform(low=0.5, high=1, size=3)
    noise_image = read(noise_image_path)
    resected_brain = blend(brain, noise_image, resection_mask, sigmas)

    write(resected_brain, output_path)
    write(resection_mask, resection_mask_output_path)


def get_resection_mask(
        resectable_hemisphere_mask,
        gray_matter_mask,
        radius,
        opening_radius=None,
        method='vtk',
    ):

    if method == 'vtk':
        voxel_center = get_random_voxel(gray_matter_mask, border=False)
        center = gray_matter_mask.TransformIndexToPhysicalPoint(tuple(voxel_center))
        r, a, s = center
        center = -r, -a, s  # LPS to RAS
        reference_path = '/tmp/ref.nii'
        write(resectable_hemisphere_mask, reference_path)
        poly_data = get_noisy_sphere_poly_data(
            center,
            radius,
        )
        model_path = '/tmp/model.vtp'
        import vtk
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(model_path)
        writer.SetInputData(poly_data)
        writer.Write()
        result_path = '/tmp/result.nii'
        sphere_mask = mesh_to_volume(reference_path, model_path, result_path)
        sphere_mask = read(result_path)
    else:
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


def mesh_to_volume(reference_path, model_path, result_path):
    command = (
        '/home/fernando/opt/Slicer/Nightly/Slicer',
        '--launch',
        '/home/fernando/opt/Slicer/Nightly/lib/Slicer-4.11/cli-modules/ModelToLabelMap',
        reference_path,
        model_path,
        result_path,
    )
    command = [str(a) for a in command]
    call(command)


def add_structural_elements(mask, min_radius=None, max_radius=None, iterations=3):
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
    radii_kernels = np.random.randint(
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


def make_noise_image(image_path, parcellation_path, output_path, threshold=True):
    image = read(image_path)
    parcellation = read(parcellation_path)
    csf_mask = get_csf_mask(image, parcellation)
    image_array = sitk.GetArrayViewFromImage(image)
    csf_mask_array = sitk.GetArrayViewFromImage(csf_mask) > 0  # to bool needed
    csf_values = image_array[csf_mask_array]
    if threshold:  # remove non-CSF voxels
        otsu = filters.threshold_otsu(csf_values)
        csf_values = csf_values[csf_values < otsu]
    distribution = norm  ##
    args = distribution.fit(csf_values)
    random_variable = distribution(*args)
    noise_array = random_variable.rvs(size=image_array.shape).astype(np.float32)
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
    random_index = np.random.randint(N)
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
    import vtk
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
        radius,
        theta=None,
        phi=None,
        noise_type=None,
        octaves=None,
        smoothness=None,
        noise_amplitude_radius_ratio=None,
        ):
    # TODO: apply trsf to sphere to make it an ellipsoid
    from noise import pnoise3, snoise3
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    from vtk.numpy_interface import dataset_adapter as dsa

    theta = 64 if theta is None else theta
    phi = 64 if phi is None else phi
    noise_type = 'perlin' if noise_type is None else noise_type
    assert noise_type in 'perlin', 'simplex'
    octaves = 2 if octaves is None else octaves
    smoothness = 10 if smoothness is None else smoothness
    noise_amplitude_radius_ratio = 0.75 if noise_amplitude_radius_ratio is None else noise_amplitude_radius_ratio

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
        noise = function(x, y, z, octaves=octaves)
        perturbance = noise_amplitude * noise
        point_with_noise = point + perturbance * normal
        points_with_noise.append(point_with_noise)
    points_with_noise = np.array(points_with_noise)

    vertices = vtk.vtkPoints()
    vertices.SetData(numpy_to_vtk(points_with_noise))
    poly_data.SetPoints(vertices)

    normalFilter = vtk.vtkPolyDataNormals()
    normalFilter.AutoOrientNormalsOn()
    normalFilter.SetComputePointNormals(True)
    normalFilter.SetComputeCellNormals(True)
    normalFilter.SplittingOff()
    normalFilter.SetInputData(poly_data)
    normalFilter.Update()
    poly_data = normalFilter.GetOutput()

    return poly_data


def get_noise_sphere_image(
        reference,
        center,
        radius,
        theta=None,
        phi=None,
        noise_type=None,
        octaves=None,
        smoothness=None,
        noise_amplitude_radius_ratio=None,
        ):
    return


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
        if 'periventricular' in line: continue
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


if __name__ == "__main__":
    np.random.seed(42)
    # input_path = Path('~/Dropbox/MRI/t1.nii.gz').expanduser()
    # # brain_segmentation_path = Path('~/Dropbox/MRI/t1_brain_seg.nii.gz').expanduser()
    # parcellation_path = Path('~/Dropbox/MRI/t1_seg_gif.nii.gz').expanduser()
    # input_path = '/home/fernando/episurg/mri/mni_alejandro/MNI_152_mri.nii.gz'
    # parcellation_path = '/home/fernando/episurg/mri/mni_alejandro/MNI_152_gif.nii.gz'

    output_dir = Path('/tmp/resected_perlin')
    output_dir.mkdir(exist_ok=True)
    # if output_dir.is_dir():
    #     import shutil
    #     shutil.rmtree(output_dir)

    N = 10

    input_path = '/tmp/1395_unbiased.nii.gz'
    parcellation_path = '/home/fernando/episurg_old/subjects/1395/mri/t1_pre/assessors/1395_t1_pre_gif_seg.nii.gz'
    noise_image_path = Path('/tmp/1395_unbiased_noise.nii.gz')
    if not noise_image_path.is_file():
        make_noise_image(input_path, parcellation_path, noise_image_path)
    for i in trange(N):
        output_path = output_dir / f'1395_resected_{i}.nii.gz'
        resection_mask_output_path = output_dir / f'1395_resected_{i}_label.nii.gz'
        hemisphere = 'left' if np.random.rand() > 0.5 else 'right'
        opening_radius = 3
        resect(
            input_path,
            parcellation_path,
            noise_image_path,
            output_path,
            resection_mask_output_path,
            hemisphere,
            opening_radius=opening_radius,
        )












    # sphere_path = '/tmp/noisy_sphere.vtp'
    # center = -37, 38, -24
    # radius = 30
    # pd = get_noisy_sphere_poly_data(
    #     center,
    #     radius,
    # )
    # import vtk
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(sphere_path)
    # writer.SetInputData(pd)
    # writer.Write()
    # # m = vtk.vtkVoxelModeller()
    # # m.SetScalarTypeToChar()
    # # m.SetInputData(pd)
    # # output = m.GetOutput()

    # ref = read(input_path)
    # white_image = vtk.vtkImageData()
    # white_image.SetSpacing(ref.GetSpacing())
    # white_image.SetDimensions(ref.GetSize())
    # white_image.SetOrigin(ref.GetOrigin())

    # pol2stenc = vtk.vtkPolyDataToImageStencil()
    # pol2stenc.SetInputData(pd)
    # pol2stenc.SetOutputOrigin(ref.GetOrigin())
    # pol2stenc.SetOutputSpacing(ref.GetSpacing())
    # extent = list(zip((0, 0, 0), ref.GetSize()))
    # extent = np.array(extent).flatten()
    # pol2stenc.SetOutputWholeExtent(extent)
    # pol2stenc.Update()

    # import itk
    # Dimension = 2
    # PixelType = itk.UC
    # ImageType = itk.Image[PixelType, Dimension]
    # vtkToItkFilter = itk.VTKImageToImageFilter[ImageType].New()
    # vtkToItkFilter.SetInput(pol2stenc.GetOutput())

    # vtkToItkFilter.Update()
    # myitkImage = vtkToItkFilter.GetOutput()
    # myitkImage.SetDirection(ref.GetDirection())
