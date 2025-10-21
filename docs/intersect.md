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
```

!!! tip

    Use `--link-type soft` to create symbolic links or `--link-type hard` to copy the files instead.

### Output Files

The following outputs will be written to the `intersection/` directory:

| File | Description |
|------|--------------|
| `intersection.parquet` | Metadata table containing information for all intersected micrographs |
| `files_in_common.txt` | List of intersected file names |
| `files` | The `.webp` file for the actual intersected micrographs (either symlinked or copied) |

This subset represents the most consistent and high-quality micrographs across both domains, suitable for subsequent refinement or reconstruction workflows.