# `resector`

<p align="center">
    <img src="https://raw.githubusercontent.com/fepegar/resector/master/docs/images/60_examples_resized_50.gif" alt="Resections">
</p>

Implementation of a [TorchIO](https://torchio.readthedocs.io/) transform
used to simulate a resection cavity from a T1-weighted brain MRI and a
corresponding [geodesic information flows (GIF) brain parcellation (version 3.0)](http://niftyweb.cs.ucl.ac.uk/program.php?p=GIF).

The corresponding talk at MICCAI 2020 is available on YouTube:

[![MICCAI 2020 - Fernando Pérez-García - Simulation of resection cavity for self-supervised learning](https://img.youtube.com/vi/RPKwlOw4r0Q/0.jpg)](https://www.youtube.com/watch?v=RPKwlOw4r0Q)

## Credit

If you use this library for your research, please cite the following publications:

[Pérez-García, F., Dorent, R., Rizzi, M., Cardinale, F., Frazzini, V., Navarro, V., Essert, C., Ollivier, I., Vercauteren, T., Sparks, R., Duncan, J.S., Ourselin, S.: *A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections*. International Journal of Computer Assisted Radiology and Surgery  – **IJCARS** (Jun 2021)](https://doi.org/10.1007/s11548-021-02420-2)

[Pérez-García, F., Rodionov, R., Alim-Marvasti, A., Sparks, R., Duncan, J.S., Ourselin, S.: *Simulation of Brain Resection for Cavity Segmentation Using Selfsupervised and Semi-supervised Learning*. In: Medical Image Computing and Computer Assisted Intervention – **MICCAI 2020**. pp. 115–125. Lecture Notes in Computer Science, Springer International Publishing, Cham (2020)](https://doi.org/10.1007/978-3-030-59716-0_12)

Bibtex:

```bibtex
@inproceedings{perez-garcia_simulation_2020,
    address = {Cham},
    series = {Lecture {Notes} in {Computer} {Science}},
    title = {Simulation of {Brain} {Resection} for {Cavity} {Segmentation} {Using} {Self}-supervised and {Semi}-supervised {Learning}},
    isbn = {978-3-030-59716-0},
    doi = {10.1007/978-3-030-59716-0\_12},
    language = {en},
    booktitle = {Medical {Image} {Computing} and {Computer} {Assisted} {Intervention} {\textendash} {MICCAI} 2020},
    publisher = {Springer International Publishing},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Rodionov, Roman and Alim-Marvasti, Ali and Sparks, Rachel and Duncan, John S. and Ourselin, S{\'e}bastien},
    year = {2020},
    keywords = {Segmentation, Self-supervised learning, Neurosurgery},
    pages = {115--125},
}

@article{perez-garcia_self-supervised_2021,
    title = {A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections},
    issn = {1861-6429},
    url = {https://doi.org/10.1007/s11548-021-02420-2},
    doi = {10.1007/s11548-021-02420-2},
    language = {en},
    urldate = {2021-06-14},
    journal = {International Journal of Computer Assisted Radiology and Surgery},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Dorent, Reuben and Rizzi, Michele and Cardinale, Francesco and Frazzini, Valerio and Navarro, Vincent and Essert, Caroline and Ollivier, Ir{\`e}ne and Vercauteren, Tom and Sparks, Rachel and Duncan, John S. and Ourselin, S{\'e}bastien},
    month = jun,
    year = {2021},
    file = {Springer Full Text PDF:/Users/fernando/Zotero/storage/SM9WHUB7/P{\'e}rez-Garc{\'i}a et al. - 2021 - A self-supervised learning strategy for postoperat.pdf:application/pdf},
}
```

## Installation

Using [`conda`](https://docs.conda.io/en/latest/miniconda.html) is recommended:

```shell
conda create --name resenv python=3.8 --yes && conda activate resenv
pip install light-the-torch
ltt install torch
pip install git+https://github.com/fepegar/resector
resect --help
```

## Usage

```shell
resect t1.nii.gz gif_parcellation.nii.gz t1_resected.nii.gz t1_resection_label.nii.gz
```

[TorchIO](https://torchio.readthedocs.io/), which is installed with `resector`, can be used to download some sample images:

```shell
T1=`python -c "import torchio as tio; print(tio.datasets.FPG().t1.path)"`
GIF=`python -c "import torchio as tio; print(tio.datasets.FPG().seg.path)"`
resect $T1 $GIF t1_resected.nii.gz t1_resection_label.nii.gz
```

Run `resect --help` for more options.

## Funding

This work was funded by the [Engineering and Physical Sciences Research Council (EPSRC)](https://epsrc.ukri.org/) and the [Wellcome Trust](https://wellcome.org/).

It was additionally supported by the [EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (i4health)](https://www.ucl.ac.uk/intelligent-imaging-healthcare/) and the [Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS)](https://www.ucl.ac.uk/interventional-surgical-sciences/).
