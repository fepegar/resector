"""Console script for resector."""
import sys
import time
import click
from pathlib import Path


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('parcellation-path', type=click.Path(exists=True))
@click.argument('output-image-path', type=click.Path())
@click.argument('output-label-path', type=click.Path())
@click.option('--seed', '-s', type=int)
@click.option('--min-volume', '-miv', type=int, default=500, show_default=True)
@click.option('--max-volume', '-mav', type=int, default=50000, show_default=True)
@click.option('--volumes-path', '-p', type=click.Path(exists=True))
@click.option('--simplex-path', '-n', type=click.Path(exists=True))
@click.option('--std-blur', type=float)
@click.option('--shape', type=click.Choice(['ellipsoid', 'cuboid', 'noisy']), default='noisy', show_default=True)
@click.option('--texture', type=click.Choice(['dark', 'random', 'csf']), default='csf', show_default=True)
@click.option('--center-ras', '-r', nargs=3, type=float)
@click.option('--wm-lesion/--no-wm-lesion', '-w', type=bool, default=False, show_default=True)
@click.option('--clot/--no-clot', '-b', type=bool, default=False, show_default=True)
@click.option('--verbose/--no-verbose', '-v', type=bool, default=False, show_default=True)
@click.option('--debug-dir', '-d', type=click.Path(file_okay=False))
@click.option('--cleanup/--no-cleanup', '-c', default=False, show_default=True)
def main(
        input_path,
        parcellation_path,
        output_image_path,
        output_label_path,
        seed,
        min_volume,
        max_volume,
        volumes_path,
        simplex_path,
        std_blur,
        shape,
        texture,
        center_ras,
        wm_lesion,
        clot,
        verbose,
        debug_dir,
        cleanup,
        ):
    import torchio as tio
    import resector

    if seed is not None:
        import torch
        torch.manual_seed(seed)

    if debug_dir is not None:
        resector.io.debug_dir = Path(debug_dir).expanduser().absolute()

    resectable_paths, gm_paths, noise_path, existed = ensure_images(
        input_path,
        parcellation_path,
    )

    try:
        if volumes_path is not None:
            import pandas as pd
            df = pd.read_csv(volumes_path)
            volumes = df.Volume.values
            kwargs = dict(volumes=volumes)
        else:
            kwargs = dict(volumes_range=(min_volume, max_volume))
        if std_blur is not None:
            kwargs['sigmas_range'] = std_blur, std_blur
        kwargs['simplex_path'] = simplex_path
        kwargs['wm_lesion_p'] = wm_lesion
        kwargs['clot_p'] = clot
        kwargs['verbose'] = verbose
        kwargs['shape'] = shape
        kwargs['texture'] = texture
        kwargs['center_ras'] = center_ras

        transform = tio.Compose((
            tio.ToCanonical(),
            resector.RandomResection(**kwargs),
        ))
        subject = tio.Subject(
            image=tio.ScalarImage(input_path),
            resection_resectable_left=tio.LabelMap(resectable_paths[0]),
            resection_resectable_right=tio.LabelMap(resectable_paths[1]),
            resection_gray_matter_left=tio.LabelMap(gm_paths[0]),
            resection_gray_matter_right=tio.LabelMap(gm_paths[1]),
            resection_noise=tio.ScalarImage(noise_path),
        )
        with resector.timer('RandomResection', verbose):
            transformed = transform(subject)
        with resector.timer('Saving images', verbose):
            transformed['image'].save(output_image_path)
            transformed['label'].save(output_label_path)
        return_code = 0
    except Exception as e:
        return_code = 1
        raise
    finally:
        if not existed and cleanup:
            with resector.timer('Cleaning up', verbose):
                for p in resectable_paths: p.unlink()
                for p in gm_paths: p.unlink()
                noise_path.unlink()
    return return_code


def ensure_images(input_path, parcellation_path):
    import resector
    hemispheres = 'left', 'right'
    input_path = Path(input_path)
    output_dir = input_path.parent
    stem = input_path.name.split('.')[0]

    existed = True
    gm_paths = []
    resectable_paths = []
    for hemisphere in hemispheres:
        dst = output_dir / f'{stem}_gray_matter_{hemisphere}_seg.nii.gz'
        gm_paths.append(dst)
        if not dst.is_file():
            existed = False
            gm = resector.parcellation.get_gray_matter_mask(
                parcellation_path, hemisphere)
            resector.io.write(gm, dst)
        dst = output_dir / f'{stem}_resectable_{hemisphere}_seg.nii.gz'
        resectable_paths.append(dst)
        if not dst.is_file():
            existed = False
            resectable = resector.parcellation.get_resectable_hemisphere_mask(
                parcellation_path,
                hemisphere,
            )
            resector.io.write(resectable, dst)
    noise_path = output_dir / f'{stem}_noise.nii.gz'
    if not noise_path.is_file():
        existed = False
        resector.parcellation.make_noise_image(
            input_path,
            parcellation_path,
            noise_path,
        )
    return resectable_paths, gm_paths, noise_path, existed


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
