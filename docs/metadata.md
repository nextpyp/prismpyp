# Evaluation of Aldolase Micrographs from EMPIAR-10379

This tutorial shows how to analyze 1,118 motion corrected micrographs of rabbit muscle aldolase from [EMPIAR-10379](https://www.ebi.ac.uk/empiar/EMPIAR-10379/).

## 1. Prepare Input Data

First, we need to create a **metadata table** used for prismPYP training and embedding generation containing information about microscope parameters, CTF statistics, and motion information for all micrographs in the dataset.

In general, you can build metadata using either **nextPYP preprocessing outputs** or **cryoSPARC outputs**, but for this example we will use pre-calculated results.

### 🧪 Download the Test Data

We will download the `example_data.tar.gz` archive from [Zenodo](https://doi.org/10.5281/zenodo.17161604), which contains micrograph and power-spectrum images plus all the necessary metadata:

```bash
mkdir example_data
tar -xvzf example_data.tar.gz -C example_data
```

This command extracts the data into an `example_data/` folder containing:

```
example_data/
    ├── pkl/                      # metadata from nextPYP preprocessing
    ├── webp/                     # 512×512 images (micrographs + power spectra)
    ├── J7_exposures_accepted_exported.cs
    ├── sp-preprocessing-*.micrographs
    └── .pyp_config.toml
```

### 📦 Intermediate Results

The [Zenodo entry](https://doi.org/10.5281/zenodo.17161604) also contains the following files:

* ```model_weights.tar.gz```: Trained model weights for the real domain (```real_model_best.pth.tar```) and the Fourier domain (```fft_model_best.pth.tar```) inputs.
* ```fft_good_export.parquet```: Data points that have high-quality features in the Fourier domain.
* ```real_good_export.parquet```: Data points that have high-qualtiy features in the real domain.

By taking the intersection between ```fft_good_export.parquet``` and ```real_good_export.parquet```, you can obtain the 862 high-quality micrographs that we used to obtain a 2.9&nbsp;Å structure of aldolase.

## 2. Build Metadata Table

Before starting, create directories to store the metadata and all generated outputs:

```bash
mkdir -p metadata
mkdir -p output_dir
```

=== "nextPYP"

      - Run the following command to assemble metadata from nextPYP preprocessing results:
         ```bash
         prismpyp metadata_nextpyp \
            --pkl-path example_data/pkl \
            --output-dir metadata \
            --cryosparc-path example_data/J7_exposures_accepted_exported.cs
         ```

      You can omit `--cryosparc-path` if you do not need **relative ice thickness** visualization.

=== "cryoSPARC"

      To build metadata directly from **cryoSPARC** outputs, you’ll need data from the `Import`, `Manually Curate Exposures` (with outputs of `Patch CTF Estimation` as the inputted micrographs), and `CTFFIND4` jobs.

      For the test dataset (EMPIAR-10379), the deposited data already contains motion corrected micrographs, so you can skip motion correction.

      - Export the outputs of the following jobs and note their locations:

         - **Import Micrographs** → `J1`
         - **Patch CTF Estimation** → `J2`
         - **CTFFIND4** → `J3`
         - **Manually Curate Exposures` → `J4`
         - cryoSPARC project directory → `/cryosparc/output/dir`

      - Build the metadata table:
         ```bash
         prismpyp metadata_cryosparc \
            --imported-dir "/cryosparc/output/dir/J1/imported" \
            --patch-ctf-file "/cryosparc/output/dir/exports/groups/J4_exposures_accepted/J4_exposures_accepted_exported.cs" \
            --ctffind-dir "/cryosparc/output/dir/J3/ctffind_output" \
            --ctffind-file "/cryosparc/output/dir/exports/groups J3_exposures_success/J3_exposures_success_exported.cs" \
            --output-dir metadata
         ```

      *Depending on how many micrographs you have, this process may take several minutes to run.*

## 3. Generated Outputs

The metadata-building command will produce a file named `micrograph_metadata.csv`, containing:

| Column | Description |
|---------|--------------|
| `micrograph_name` | Name of each micrograph |
| `rel_ice_thickness` | Relative ice thickness (if `--cryosparc-path` was provided) |
| `ctf_fit` | CTF fit correlation coefficient |
| `est_resolution` | Estimated resolution in Å |
| `avg_motion` | Average beam-induced motion |
| `num_particles` | Number of picked particles |
| `mean_defocus` | Mean defocus (Å) |

In addition, the following files are generated:

- `pixel_size.txt` — microscope pixel size for this dataset  
- `all_micrographs_list.micrographs` — list of all micrographs (without extensions)  
- `webp/` — directory of `.webp` images for both micrographs and their CTFFIND4-derived power spectra
