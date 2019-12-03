import numpy as np
import nibabel as nib
from tqdm import tqdm
from noise import pnoise3, snoise3


offset = 1000
octaves = 4


side_world = 4
spacing = 0.04
side_voxels = int(side_world / spacing)
si = sj = sk = side_voxels
shape = si, sj, sk

array = np.empty(shape, dtype=np.float32)

affine = np.diag((spacing, spacing, spacing, 1))
origin = - (side_voxels/2 - 0.5) * spacing
origin = 3 * (origin,)
affine[:3, 3] = origin

grid = np.meshgrid(
    np.arange(si),
    np.arange(sj),
    np.arange(sk),
    # sparse=True,
)

indices = np.array(grid).T.reshape(-1, 3)
points = nib.affines.apply_affine(affine, indices) + offset
zipped = list(zip(indices, points))
for index, point in tqdm(zipped):
    noise = snoise3(*point, octaves=octaves)
    i, j, k = index
    array[i, j, k] = noise

nii = nib.Nifti1Image(array, affine)
nii.header['qform_code'] = 1
nii.header['sform_code'] = 1
nii.to_filename('/tmp/simplex.nii.gz')
