# -*- coding: utf-8 -*-

"""Console script for resector."""
import sys
import click
from pathlib import Path


@click.command()
@click.argument('output-path', type=click.Path())
@click.option('--size', '-s', type=int, default=256, show_default=True)
@click.option('--noise-offset', '-o', type=float, default=1000, show_default=True)
@click.option('--noise-scale', '-s', type=float, default=0.02, show_default=True)
@click.option('--min-persistence', '-p', type=float, default=0.01, show_default=True)
@click.option('--max-persistence', '-g', type=float, default=0.8, show_default=True)
def main(
        output_path,
        size,
        noise_offset,
        noise_scale,
        min_persistence,
        max_persistence,
        ):
    import numpy as np
    import nibabel as nib
    from tqdm import trange
    from noise import snoise3

    """
    Original JavaScript code:

    let center = createVector(width/2, height/2);
    let maxd = center.mag();
    for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
            let p = createVector(j, i);
            let d = dist(p.x, p.y, center.x, center.y);
            persistence = map(d, 0, maxd, 0.01, 0.6);
            noiseDetail(octaves, persistence);
            let noiseVal = noise(noiseOffset + j * noiseScaleX, noiseOffset + i * noiseScaleY);
            noiseVal -= intensityOffset;
            noiseVal = constrain(noiseVal, 0, 1);
            let intensity = map(noiseVal, 0, 1, 0, 255);
            intensity = constrain(intensity, 0, 255);
            set(j, i, intensity);
        }
    }
    """
    output_size = si, sj, sk = 3 * [size]
    output = np.empty(output_size, np.float32)
    center = np.array(output_size) / 2
    maxd = np.linalg.norm(center)
    for i in trange(si):
        for j in range(sj):
            for k in range(sk):
                p = np.array((i, j, k))
                d = get_distance(p, center)
                persistence = map(d, 0, maxd, min_persistence, max_persistence)
                noise_val = snoise3(
                    noise_offset + k * noise_scale,
                    noise_offset + j * noise_scale,
                    noise_offset + i * noise_scale,
                    octaves=4,
                    persistence=persistence,
                )
                noise_val = (noise_val + 1) / 2  # [0, 1]
                output[i, j, k] = noise_val
    affine = np.eye(4)
    affine[:3, 3] = -center
    nii = nib.Nifti1Image(output, affine)
    nii.to_filename(output_path)
    return 0

def get_distance(a, b):
    import numpy as np
    return np.linalg.norm(a - b)

def map(n, start1, stop1, start2, stop2):
    # https://github.com/processing/p5.js/blob/b15ca7c7ac8ba95ccb0123cf74fc163040090e1b/src/math/calculation.js#L450
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
