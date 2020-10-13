# `resector`

<p align="center">
    <img src="https://raw.githubusercontent.com/fepegar/resector/master/docs/images/60_examples_resized_50.gif"
        alt="Resections">
</p>

Implementation of a [TorchIO](https://torchio.readthedocs.io/) transform
used to simulate a resection cavity from a T1-weighted brain MRI and a
corresponding [geodesic information flows (GIF) brain parcellation (version 3.0)](http://niftyweb.cs.ucl.ac.uk/program.php?p=GIF).

## Credit

If you use this library for your research, please cite our MICCAI 2020 paper:

[F. Pérez-García, R. Rodionov, A. Alim-Marvasti, R. Sparks, J. S. Duncan and
S. Ourselin. *Simulation of Brain Resection for Cavity Segmentation Using
Self-Supervised and Semi-Supervised Learning*](https://link.springer.com/chapter/10.1007%2F978-3-030-59716-0_12).

Bibtex:

```bibtex
@InProceedings{10.1007/978-3-030-59716-0_12,
    author="P{\'e}rez-Garc{\'i}a, Fernando
    and Rodionov, Roman
    and Alim-Marvasti, Ali
    and Sparks, Rachel
    and Duncan, John S.
    and Ourselin, S{\'e}bastien",
    title="Simulation of Brain Resection for Cavity Segmentation Using Self-supervised and Semi-supervised Learning",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="115--125",
    isbn="978-3-030-59716-0"
}
```

[[Preprint on arXiv](https://arxiv.org/abs/2006.15693)]

## Installation

```shell
$ git clone https://github.com/fepegar/resector.git
$ pip install --editable ./resector
```

## Usage

```shell
$ resect t1.nii.gz gif_parcellation.nii.gz t1_resected.nii.gz t1_resection_label.nii.gz
```

Run `resect --help` for more options.
