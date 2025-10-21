# Performing Dual-Domain Filtering

After selecting high-quality subsets in both the **real** and **Fourier** domains, you can intersect them to identify micrographs that exhibit strong features in both spaces.

This step combines the complementary strengths of real-space and frequency-space representations to yield the most reliable set of high-quality images for downstream analysis.

!!! tip

    Run this step on a **remote cluster** for large datasets. Transfer the `.parquet` files from your local Phoenix session to the cluster before starting.

### Activate the Environment

```bash
conda activate prismpyp
```

### Create an Output Directory

```bash
mkdir intersection
```

### Compute the Intersection

Run the following command to take the intersection between real and Fourier domain selections:

```bash
prismpyp intersect \
--parquet-files output_dir/fft/fft_good_export.parquet output_dir/real/real_good_export.parquet \
--output-folder intersection \
--link-type soft \
--webp-path metadata/webp
--mrc-path 
```

### Generate .mrc Files For Re-Uploading to CryoSPARC or nextPYP

Assuming that you have the original .mrc files that were pre-processed in either nextPYP or cryoSPARC at a path `/path/to/mrcs`, you can also use the `intersect` command to symlink or copy the subset of good micrographs as `.mrc` files for re-uploading into nextPYP or cryoSPARC by specifying the `--mrc-path` argument:

```bash
prismpyp intersect \
--parquet-files output_dir/fft/fft_good_export.parquet output_dir/real/real_good_export.parquet \
--output-folder intersection \
--link-type soft \
--webp-path metadata/webp
--mrc-path /path/to/mrcs
```

This will write the good subset's `.mrc` files to the `intersection` folder.

!!! tip

    Use `--link-type soft` to create symbolic links or `--link-type hard` to copy the files instead.

### Output Files

The following outputs will be written to the `intersection/` directory:

| File | Description |
|------|--------------|
| `intersection.parquet` | Metadata table containing information for all intersected micrographs |
| `files_in_common.txt` | List of intersected file names |
| `files` | The `.webp` file for the actual intersected micrographs (either symlinked or copied) |
| `mrcs` | The `.mrc` file for the actual intersected micrographs (either symlinked or copied) |

This subset represents the most consistent and high-quality micrographs across both domains, suitable for subsequent refinement or reconstruction workflows.