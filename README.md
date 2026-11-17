# vmif-hgsoc

Evaluation of virtual multiplex immunofluorescence (GigaTIME, GigaTIME-Flash, HEX) from H&E in high-grade serous ovarian cancer: association with measured proteins, label-efficiency against a foundation-model encoder (UNI), platinum-response prediction, and tile-level validation against same-section multiplex imaging (Orion).

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/lilab-stanford/MUSK
```

Set paths in `config.yaml`. UNI and MUSK weights are loaded from the Hugging Face Hub (access must be granted by the model owners); GigaTIME, GigaTIME-Flash and HEX (`hex_nolan48`) checkpoints are read from the paths given.

## Data layout

```
data_root/
  ptrc/slides/*.svs                       TCIA PTRC-HGSOC H&E
  ptrc/PTRC-HGSOC_List_clincal_data.xlsx  TCIA clinical table
  ptrc/proteome/*.tsv                     PDC global proteome (before imputation, imputed)
  tcga/slides/*.svs                       GDC TCGA-OV H&E
  tcga/mmc2_PlatinumStatus.xlsx           Zhang et al., Cell 2016, Table S2
  tcga/sztupinszki2021_supplementary.txt  Sztupinszki et al., Clin Cancer Res 2021, supplementary table
  tcga/pancan/*                           GDC PanCanAtlas files (leukocyte fraction, CIBERSORT, ABSOLUTE, RPPA, HRD, subtypes)
  orion/CRC*/                             Orion registered H&E (*.ome.tif) and single-cell tables (*.csv)
```

TCGA RNA-seq, U133A and CPTAC protein values are retrieved from cBioPortal.

## Run

```bash
python scripts/01_extract.py PTRC-HGSOC      # tiling, UNI/MUSK features, decoder outputs (GPU)
python scripts/01_extract.py TCGA-OV
python scripts/02_cohorts.py                 # slide tables, labels, molecular references
python scripts/03_features.py                # slide-level features for each decoder configuration
python scripts/04_association.py             # specificity, partial correlations, compartment and sensitivity analyses
python scripts/05_supervised.py              # learning curves, n*, task panel, random projections, provenance
python scripts/06_platinum.py                # attention/mean MIL, fusion, covariates, initialisation seeds (GPU)
python scripts/06_platinum.py --stage summarise
python scripts/07_spatial.py                 # spatial descriptors and coordinate-shuffle null
python scripts/08_orion.py                   # same-section validation (GPU)
```

Results are written to `work_root/results/`. All supervised preprocessing (scaling, block weighting, PCA) is fitted on training patients within each split; splits are grouped by patient.


## License

MIT (see `LICENSE`). Files in `third_party/` are copied unmodified from the original model repositories and remain under their original licenses.
