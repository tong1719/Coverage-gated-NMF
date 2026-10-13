# External data used by the experiments

No third-party raw spectra are stored in this repository.

## MassBank fragmentation profiles

Source: [MassBank-data release 2026.03](https://github.com/MassBank/MassBank-data/tree/2026.03/AAFC). The scripts use the AAFC records `AC000008`, `AC000009`, `AC000010`, `AC000015`, `AC000020`, `AC000025`, `AC000030`, `AC000035`, `AC000040`, `AC000045`, `AC000050`, and `AC000055`. `prepare_data.py` downloads the corresponding repository objects and verifies each pinned Git blob hash. Each MassBank record retains its source licence and copyright information.

## Raman oil-mixture spectra

Source: Pierre Picouet, *Raman spectra of extra virgin olive and sunflower oil mixtures*, [DOI: 10.57745/DOGT0E](https://doi.org/10.57745/DOGT0E). Obtain the original CSV from the dataset page; do not use a resaved spreadsheet export. The expected SHA-256 is:

`c426967faac3005b1145385d5f8da5bfa85078084f7c5931b30a9bf9fcb11758`

The source portal identifies the dataset licence as Licence Ouverte / Open Licence 2.0. The CSV is used for a retrospective binning experiment, not for prospective acquisition under multiple instrument settings.
