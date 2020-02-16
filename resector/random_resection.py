import copy
import enum
import torch
import numpy as np
from math import tau
import SimpleITK as sitk

from torchio import LABEL, DATA, AFFINE, IMAGE

from .io import nib_to_sitk, get_sphere_poly_data
from .resector import resect


@enum.unique
class Hemisphere(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'



class RandomResection:
    def __init__(
            self,
            volumes_range=None,
            volumes=None,
            sigmas_range=(0.5, 1),
            radii_ratio_range=(1, 2),
            angles_range=(0, 360),
            delete_resection_keys=True,
            keep_original=False,
            force_positive=True,
            add_params=True,
            add_resected_structures=False,
            seed=None,
            verbose=False,
            ):
        """
        Either volumes or volume_range should be passed
        volumes is an iterable of possible volumes (they come from EPISURG)
        volumes_range is a range for a uniform distribution (TODO: fit a distribution?)

        Assume there is a key 'image' in sample dict
        Assume there is a key 'resectable_left' in sample dict
        Assume there is a key 'resectable_right' in sample dict
        Assume there is a key 'gray_matter_left' in sample dict
        Assume there is a key 'gray_matter_right' in sample dict
        Assume there is a key 'noise' in sample dict
        """
        if (volumes is None and volumes_range is None
                or volumes is not None and volumes_range is not None):
            raise ValueError('Please enter a value for volumes or volumes_range')
        self.volumes = volumes
        self.volumes_range = volumes_range
        self.sigmas_range = sigmas_range
        self.radii_ratio_range = radii_ratio_range
        self.angles_range = angles_range
        self.delete_resection_keys = delete_resection_keys
        self.keep_original = keep_original
        self.force_positive = force_positive
        self.add_params = add_params
        self.add_resected_structures = add_resected_structures
        self.seed = seed
        self.verbose = verbose
        self.sphere_poly_data = get_sphere_poly_data()

    def __call__(self, sample):
        self.check_seed()
        if self.verbose:
            print('Sample stem for resection:', sample[IMAGE]['stem'])
            import time
            start = time.time()
        resection_params = self.get_params(
            self.volumes,
            self.volumes_range,
            self.sigmas_range,
            self.radii_ratio_range,
            self.angles_range,
        )
        # This makes my life easier for RandomMotion as some MRI have negative
        # values
        if self.force_positive:
            im_dict = sample[IMAGE]
            min_image_value = sample[IMAGE][DATA].min()
            if min_image_value < 0:
                if self.verbose:
                    print('Forcing positive values on', sample[IMAGE]['stem'])
                noise_dict = sample['resection_noise']
                im_dict[DATA] = im_dict[DATA] - min_image_value
                noise_dict[DATA] = noise_dict[DATA] - min_image_value
        t1_pre = nib_to_sitk(
            sample[IMAGE][DATA][0],
            sample[IMAGE][AFFINE],
        )
        hemisphere = resection_params['hemisphere']
        gray_matter_mask = nib_to_sitk(
            sample[f'resection_gray_matter_{hemisphere}'][DATA][0],
            sample[f'resection_gray_matter_{hemisphere}'][AFFINE],
        )
        resectable_hemisphere_mask = nib_to_sitk(
            sample[f'resection_resectable_{hemisphere}'][DATA][0],
            sample[f'resection_resectable_{hemisphere}'][AFFINE],
        )
        noise_image = nib_to_sitk(
            sample['resection_noise'][DATA][0],
            sample['resection_noise'][AFFINE],
        )
        if self.verbose:
            duration = time.time() - start
            print(f'[Prepare resection images]: {duration:.1f} seconds')

        resected_brain, resection_mask, resection_center = resect(
            self.sphere_poly_data,
            t1_pre,
            gray_matter_mask,
            resectable_hemisphere_mask,
            noise_image,
            resection_params['sigmas'],
            resection_params['radii'],
            resection_params['angles'],
            resection_params['noise_offset'],
            verbose=self.verbose,
        )
        resection_params['resection_center'] = resection_center
        resected_brain_array = self.sitk_to_array(resected_brain)
        resected_mask_array = self.sitk_to_array(resection_mask)
        image_resected = self.add_channels_axis(resected_brain_array)
        resection_label = self.add_channels_axis(resected_mask_array)
        resection_label = resection_label.astype(np.float32)
        assert image_resected.ndim == 4
        assert resection_label.ndim == 4

        ## Update sample
        if self.delete_resection_keys:
            del sample['resection_gray_matter_left']
            del sample['resection_gray_matter_right']
            del sample['resection_resectable_left']
            del sample['resection_resectable_right']
            del sample['resection_noise']

        # Add resected image and label to sample
        if self.add_params:
            sample['random_resection'] = resection_params
        if self.keep_original:
            sample['image_original'] = copy.deepcopy(sample[IMAGE])
        sample[IMAGE][DATA] = torch.from_numpy(image_resected)
        label_dict = dict(
            data=torch.from_numpy(resection_label),
            affine=sample[IMAGE]['affine'],
            stem=sample[IMAGE]['stem'],
            type=LABEL,
            path='',
        )
        sample['label'] = label_dict

        if self.add_resected_structures:
            sample['resected_structures'] = self.get_resected_structures(
                sample, resection_mask)

        if self.verbose:
            duration = time.time() - start
            print(f'RandomResection: {duration:.1f} seconds')
        return sample

    @staticmethod
    def get_params(
            volumes,
            volumes_range,
            sigmas_range,
            radii_ratio_range,
            angles_range,
        ):
        # Hemisphere
        hemisphere = Hemisphere.LEFT if RandomResection.flip_coin() else Hemisphere.RIGHT

        # Equivalent sphere volume
        if volumes is None:
            volume = torch.FloatTensor(1).uniform_(*volumes_range).item()
        else:
            index = torch.randint(len(volumes), (1,)).item()
            volume = volumes[index]

        # Sigmas for mask gaussian blur
        sigmas = torch.FloatTensor(3).uniform_(*sigmas_range).tolist()

        # Ratio between two of the radii of the ellipsoid
        radii_ratio = torch.FloatTensor(1).uniform_(*radii_ratio_range).item()

        # As the center of the sphere lies at the border of the brain,
        # volume should be the volume of a hemisphere. We ignore this to
        # take into account gray matter that is in deep sulci and to
        # create smaller resections, i.e. harder cases
        radius = (3 / 2 * volume / tau)**(1 / 3)
        a = radius
        b = a * radii_ratio
        c = radius ** 3 / (a * b)  # a * b * c = r**3
        radii = a, b, c

        # Rotation angles of the ellipsoid
        angles = torch.FloatTensor(3).uniform_(*angles_range).tolist()

        # Offset for noise
        noise_offset = torch.randint(1000, (1,)).item()

        parameters = dict(
            hemisphere=hemisphere.value,
            volume=volume,
            sigmas=sigmas,
            angles=angles,
            radii=radii,
            noise_offset=noise_offset,
        )
        return parameters

    def get_resected_structures(self, sample, resection_mask):
        from utils import AffineMatrix, sglob
        from tempfile import NamedTemporaryFile
        from episurg.parcellation import GIFParcellation
        mni_path = sample[IMAGE]['path']
        mni_dir = mni_path.parent
        dataset_dir = mni_dir.parent
        parcellation_dir = dataset_dir / 'parcellation'
        stem = mni_path.name.split('_t1_pre')[0]
        transform_path = sglob(mni_dir, f'{stem}*.txt')
        parcellation_path = sglob(parcellation_dir, f'{stem}*.nii.gz')
        transform = AffineMatrix(transform_path).get_itk_transform()
        parcellation = sitk.ReadImage(str(parcellation_path))
        resampled = sitk.Resample(
            parcellation,
            resection_mask,
            transform,
            sitk.sitkNearestNeighbor,
        )
        with NamedTemporaryFile(suffix='.nii') as p:
            with NamedTemporaryFile(suffix='.nii') as m:
                parcellation_path = p.name
                mask_path = m.name
                sitk.WriteImage(resampled, parcellation_path)
                sitk.WriteImage(resection_mask, mask_path)
                parcellation = GIFParcellation(parcellation_path)
                resected_structures = parcellation.get_resected_ratios(
                    mask_path)
        return resected_structures


    def check_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)

    @staticmethod
    def flip_coin():
        return torch.rand(1) >= 0.5

    @staticmethod
    def sitk_to_array(image):
        array = sitk.GetArrayFromImage(image)
        return array.transpose(2, 1, 0)

    @staticmethod
    def add_channels_axis(array):
        return array[np.newaxis, ...]

    @staticmethod
    def add_background_channel(foreground):
        background = 1 - foreground
        return np.stack((background, foreground))
