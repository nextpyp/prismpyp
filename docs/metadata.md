# Preparing Input Data and Building Metadata

This section describes how to prepare the **metadata table** used for prismPYP training and embedding generation.  

You can build metadata using either **nextPYP preprocessing outputs** or **CryoSPARC outputs**.

> 💡 The resulting metadata table consolidates microscope parameters, CTF statistics, and motion information across all micrographs in your dataset.

## 1. Test Data and Results

### 🧪 Download the Test Data

You can test prismPYP using the **`example_data.tar.gz`** archive from [Zenodo](https://doi.org/10.5281/zenodo.17161604),  
which contains micrograph and power-spectrum images plus metadata from EMPIAR-10379.

```bash
mkdir example_data
tar -xvzf example_data.tar.gz -C example_data
```

This command extracts the data into an example_data/ folder containing:

```
example_data/
    ├── pkl/                      # metadata from nextPYP preprocessing
    ├── webp/                     # 512×512 images (micrographs + power spectra)
    ├── J7_exposures_accepted_exported.cs
    ├── sp-preprocessing-*.micrographs
    └── .pyp_config.toml
```

### 📦 Intermediate Results

The [Zenodo link](https://doi.org/10.5281/zenodo.17161604) also contains the following files:

* ```model_weights.tar.gz```: Trained model weights for the real domain input (```real_model_best.pth.tar```) and the Fourier domain input (```fft_model_best.pth.tar```)
* ```fft_good_export.parquet```: Data points that have high-quality features in the Fourier domain
* ```real_good_export.parquet```: Data points that have high-qualtiy features in the real domain

By taking the intersection of ```fft_good_export.parquet``` and ```real_good_export.parquet```, you can obtain the 862 micrographs that were used to obtain the 2.9&nbsp;Å structure in the paper.

## 2. Build the Metadata Table

Before starting, create a directory to store all generated outputs:

```bash
mkdir -p output_dir
```

=== "nextPYP"

   a. Create an output directory for nextPYP-derived metadata:
   ```bash
   mkdir -p metadata_from_nextpyp
   ```

   b. Run the following command to assemble metadata from nextPYP preprocessing results:
   ```bash
   prismpyp metadata_nextpyp \
      --pkl-path example_data/pkl \
      --output-dir metadata_from_nextpyp \
      --cryosparc-path example_data/J7_exposures_accepted_exported.cs
   ```

   > You can omit `--cryosparc-path` if you do not need **relative ice thickness** visualization.

=== "cryoSPARC"

   To build metadata directly from **CryoSPARC** outputs, you’ll need data from the `Import`, `Patch CTF Estimation`, and `CTFFIND4` jobs.

   > For the test dataset (EMPIAR-10379), the deposited data already contains aligned micrographs, so you can skip motion correction.

   a. Export the outputs of the following jobs and note their locations:
      - **Import Micrographs** → `J1`
      - **Patch CTF Estimation** → `J2`
      - **CTFFIND4** → `J3`
      - CryoSPARC project directory → `/cryosparc/output/dir`

   b. Create the metadata directory:
      ```bash
      mkdir -p metadata_from_cryosparc
      ```

   c. Build the metadata table:
      ```bash
      prismpyp metadata_cryosparc \
         --imported-dir "/cryosparc/output/dir/J1/imported" \
         --patch-ctf-file "/cryosparc/output/dir/J2/J2_passthrough_exposures_accepted.cs" \
         --ctffind-dir "/cryosparc/output/dir/J3/ctffind_output" \
         --ctffind-file "/cryosparc/output/dir/exports/groups J3_exposures_success/J3_exposures_success_exported.cs" \
         --output-dir metadata_from_cryosparc
      ```

## 3. Outputs

Both metadata-building commands will produce a file named `micrograph_metadata.csv`, containing:

| Column | Description |
|---------|--------------|
| `micrograph_name` | Name of each micrograph |
| `rel_ice_thickness` | Relative ice thickness (if `--cryosparc-path` is provided) |
| `ctf_fit` | CTF fit correlation coefficient |
| `est_resolution` | Estimated resolution in Å |
| `avg_motion` | Average beam-induced motion |
| `num_particles` | Number of picked particles |
| `mean_defocus` | Mean defocus (Å) |

In addition, the following files are generated:

- `pixel_size.txt` — microscope pixel size for this dataset  
- `all_micrographs_list.micrographs` — list of all micrographs (no extensions)  
- `webp/` — directory of `.webp` images for both micrographs and their CTFFIND4-derived power spectra
<!-- 
> For the remainder of this tutorial, we’ll assume you’re using the `metadata_from_nextpyp` directory.  
> You can easily switch to another dataset by setting `--metadata-path` to `metadata_from_nextpyp` or `metadata_from_cryosparc`, depending on your source. -->
