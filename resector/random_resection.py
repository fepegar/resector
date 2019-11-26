import enum
import torch
import numpy as np
import SimpleITK as sitk

from .resector import _resect


class Hemisphere(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'


class RandomResection:
    def __init__(self, volumes):
        # From episurg dataset (although it looks bimodal)
        self.volumes = volumes
        self.sigmas_range = 0.5, 1

    def __call__(self, sample):
        """
        Assume there is a single channel in sample['image']
        Assume there is a key 'resectable_left' in sample dict
        Assume there is a key 'resectable_right' in sample dict
        Assume there is a key 'gray_matter_left' in sample dict
        Assume there is a key 'gray_matter_right' in sample dict
        Assume there is a key 'noise' in sample dict
        """
        resection_params = self.get_params(
            self.volumes,
            self.sigmas_range,
        )
        brain = self.nib_to_sitk(sample['image'][0], sample['affine'])
        hemisphere = resection_params['hemisphere']
        gray_matter_mask = self.nib_to_sitk(
            sample[f'gray_matter_{hemisphere}'], sample['affine'])
        resectable_hemisphere_mask = self.nib_to_sitk(
            sample[f'resectable_{hemisphere}'], sample['affine'])
        noise_image = self.nib_to_sitk(
            sample['noise'], sample['affine'])

        resected_brain, resection_mask, resection_center = _resect(
            brain,
            gray_matter_mask,
            resectable_hemisphere_mask,
            noise_image,
            resection_params['volume'],
            resection_params['sigmas'],
        )
        resection_params['resection_center'] = resection_center
        resected_brain_array = self.sitk_to_array(resected_brain)
        resected_mask_array = self.sitk_to_array(resection_mask)
        image_resected = self.add_channels_axis(resected_brain_array)
        resection_label = self.add_background_channel(resected_mask_array)
        assert image_resected.ndim == 4
        assert resection_label.ndim == 4

        # Update sample
        sample['random_resection'] = resection_params
        sample['image'] = image_resected
        sample['label'] = resection_label
        return sample

    @staticmethod
    def get_params(
            volumes,
            sigmas_range,
        ):

        # Hemisphere
        hemisphere = Hemisphere.LEFT if RandomResection.flip_coin() else Hemisphere.RIGHT

        # Equivalent sphere volume
        volume = torch.FloatTensor(1).uniform_(*volumes).item()

        # Sigmas for mask gaussian blur
        sigmas = torch.FloatTensor(3).uniform_(*sigmas_range).tolist()

        parameters = dict(
            hemisphere=hemisphere.value,
            volume=volume,
            sigmas=sigmas,
        )
        return parameters

    @staticmethod
    def flip_coin():
        return torch.rand(1) >= 0.5

    @staticmethod
    def nib_to_sitk(array, affine):
        """
        TODO: figure out how to get directions from affine
        so that I don't need this
        """
        import nibabel as nib
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix='.nii') as f:
            nib.Nifti1Image(array, affine).to_filename(f.name)
            image = sitk.ReadImage(f.name)
        return image

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
