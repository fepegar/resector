import copy
import enum
import torch
import numpy as np
from math import tau
import SimpleITK as sitk

import torchio as tio
from torchio import IMAGE

from .timer import timer
from .resector import resect
from .io import get_sphere_poly_data


@enum.unique
class Hemisphere(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'


class RandomResection:
    def __init__(
            self,
            image_name=IMAGE,
            volumes_range=None,
            volumes=None,
            sigmas_range=(0.5, 1),
            radii_ratio_range=(1, 2),
            angles_range=(0, 360),
            delete_resection_keys=True,
            keep_original=False,
            add_params=True,
            add_resected_structures=False,
            simplex_path=None,
            wm_lesion_p=1,
            clot_p=1,
            shape='noisy',
            texture='csf',
            center_ras=None,
            verbose=False,
            ):
        """
        Either volumes or volume_range should be passed
        volumes is an iterable of possible volumes (they come from EPISURG)
        volumes_range is a range for a uniform distribution
        (TODO: fit a distribution?)

        Assume there is a key 'image' in sample dict
        Assume there is a key 'resection_resectable_left' in sample dict
        Assume there is a key 'resection_resectable_right' in sample dict
        Assume there is a key 'resection_gray_matter_left' in sample dict
        Assume there is a key 'resection_gray_matter_right' in sample dict
        Assume there is a key 'resection_noise' in sample dict
        """
        if (volumes is None and volumes_range is None
                or volumes is not None and volumes_range is not None):
            raise ValueError('Enter a value for volumes or volumes_range')
        self.volumes = volumes
        self.volumes_range = volumes_range
        self.image_name = image_name
        self.sigmas_range = sigmas_range
        self.radii_ratio_range = radii_ratio_range
        self.angles_range = angles_range
        self.delete_resection_keys = delete_resection_keys
        self.keep_original = keep_original
        self.add_params = add_params
        self.add_resected_structures = add_resected_structures
        self.sphere_poly_data = get_sphere_poly_data()
        self.simplex_path = simplex_path
        self.shape = shape
        self.texture = texture
        self.wm_lesion_p = wm_lesion_p
        self.clot_p = clot_p
        self.center_ras = center_ras
        self.verbose = verbose

    def __call__(self, subject: tio.Subject):
        # Sampler random parameters
        resection_params = self.get_params(
            self.volumes,
            self.volumes_range,
            self.sigmas_range,
            self.radii_ratio_range,
            self.angles_range,
            self.wm_lesion_p,
            self.clot_p,
        )
        # Convert images to SimpleITK
        with timer('Convert to SITK', self.verbose):
            t1_pre = subject[self.image_name].as_sitk()
            hemisphere = resection_params['hemisphere']
            gm_name = f'resection_gray_matter_{hemisphere}'
            gray_matter_image = subject[gm_name]
            gray_matter_mask = gray_matter_image.as_sitk()
            resectable_name = f'resection_resectable_{hemisphere}'
            resectable_tissue_image = subject[resectable_name]
            resectable_tissue_mask = resectable_tissue_image.as_sitk()

            add_wm = resection_params['add_wm_lesion']
            add_clot = resection_params['add_clot']
            use_csf_image = self.texture == 'csf' or add_wm or add_clot
            if use_csf_image:
                noise_image = subject['resection_noise'].as_sitk()
            else:
                noise_image = None

        # Simulate resection
        with timer('Resection', self.verbose):
            results = resect(
                t1_pre,
                gray_matter_mask,
                resectable_tissue_mask,
                resection_params['sigmas'],
                resection_params['radii'],
                noise_image=noise_image,
                shape=self.shape,
                texture=self.texture,
                angles=resection_params['angles'],
                noise_offset=resection_params['noise_offset'],
                sphere_poly_data=self.sphere_poly_data,
                wm_lesion=add_wm,
                clot=add_clot,
                simplex_path=self.simplex_path,
                center_ras=self.center_ras,
                verbose=self.verbose,
            )
        resected_brain, resection_mask, resection_center, clot_center = results

        # Store centers for visualization purposes
        resection_params['resection_center'] = resection_center
        resection_params['clot_center'] = clot_center

        # Convert from SITK
        with timer('Convert from SITK', self.verbose):
            resected_brain_array = self.sitk_to_array(resected_brain)
            resected_mask_array = self.sitk_to_array(resection_mask)
            image_resected = self.add_channels_axis(resected_brain_array)
            resection_label = self.add_channels_axis(resected_mask_array)
        assert image_resected.ndim == 4
        assert resection_label.ndim == 4

        # Update subject
        if self.delete_resection_keys:
            subject.remove_image('resection_gray_matter_left')
            subject.remove_image('resection_gray_matter_right')
            subject.remove_image('resection_resectable_left')
            subject.remove_image('resection_resectable_right')
            if use_csf_image:
                subject.remove_image('resection_noise')

        # Add resected image and label to subject
        if self.add_params:
            subject['random_resection'] = resection_params
        if self.keep_original:
            subject['image_original'] = copy.deepcopy(subject[self.image_name])
        subject[self.image_name].data = torch.from_numpy(image_resected)
        label = tio.LabelMap(
            tensor=resection_label,
            affine=subject[self.image_name].affine,
        )
        subject.add_image(label, 'label')

        if self.add_resected_structures:
            subject['resected_structures'] = self.get_resected_structures(
                subject, resection_mask)

        return subject

    def get_params(
            self,
            volumes,
            volumes_range,
            sigmas_range,
            radii_ratio_range,
            angles_range,
            wm_lesion_p,
            clot_p,
            ):
        # Hemisphere
        if not self.center_ras or self.center_ras is None:
            hemisphere = Hemisphere.LEFT if self.flip_coin() else Hemisphere.RIGHT
        else:
            # Assuming brain in MNI space
            hemisphere = Hemisphere.LEFT if self.center_ras[0] < 0 else Hemisphere.RIGHT

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

        # Add white matter lesion
        add_wm_lesion = wm_lesion_p > torch.rand(1)

        # Add blood clot
        add_clot = clot_p > torch.rand(1)

        parameters = dict(
            hemisphere=hemisphere.value,
            volume=volume,
            sigmas=sigmas,
            angles=angles,
            radii=radii,
            noise_offset=noise_offset,
            add_wm_lesion=add_wm_lesion,
            add_clot=add_clot,
        )
        return parameters

    def get_resected_structures(self, sample, resection_mask):
        from pathlib import Path
        from tempfile import NamedTemporaryFile
        from utils import AffineMatrix, sglob
        from episurg.parcellation import GIFParcellation
        mni_path = Path(sample[self.image_name]['path'])
        mni_dir = mni_path.parent
        dataset_dir = mni_dir.parent
        parcellation_dir = dataset_dir / 'parcellation'
        stem = mni_path.name.split('_t1_pre')[0]
        transform_path = sglob(mni_dir, f'{stem}*.txt')[0]
        parcellation_path = sglob(parcellation_dir, f'{stem}*.nii.gz')[0]
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
