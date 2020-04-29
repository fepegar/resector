from pathlib import Path
import pandas as pd
from torchio import Image, ImagesDataset, INTENSITY, Subject
from resector import RandomResection

images_dir = Path('/tmp/noise')
subject_id = '1423'

def gp(s):
    return f'{subject_id}_t1_pre_{s}.nii.gz'

subject = Subject(
    Image('image', images_dir / gp('on_mni'), INTENSITY),
    Image('resection_noise', images_dir / gp('noise'), None),
    Image('resection_gray_matter_left', images_dir / gp('gray_matter_left_seg'), None),
    Image('resection_resectable_left', images_dir / gp('resectable_left_seg'), None),
    Image('resection_gray_matter_right', images_dir / gp('gray_matter_right_seg'), None),
    Image('resection_resectable_right', images_dir / gp('resectable_right_seg'), None),
)

df_volumes = pd.read_csv(Path('~/episurg/volumes.csv').expanduser())
volumes = df_volumes.Volume.values

transform = RandomResection(
    volumes=volumes,
    # sigmas_range=(0.75, 0.75),
    keep_original=True,
    verbose=True,
    # seed=42,
)

dataset = ImagesDataset([subject])
transformed = dataset[0]

for i in range(10):
    transformed = transform(dataset[0])
    dataset.save_sample(
        transformed,
        dict(
            image=f'/tmp/resected_{i}.nii.gz',
            # image_original='/tmp/resected_original.nii.gz',
            label=f'/tmp/resected_label_{i}.nii.gz',
        ),
    )
