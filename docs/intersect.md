# 🔍 Performing Dual-Domain Filtering

After selecting high-quality subsets in both the **real** and **Fourier** domains, you can intersect them to identify micrographs that exhibit strong features in both spaces.

This step combines the complementary strengths of real-space and frequency-space representations to yield the most reliable set of high-quality images for downstream analysis.

> 💡 Run this step on a **remote cluster** for large datasets. Transfer the `.parquet` files from your local Phoenix session to the cluster before starting.

---

## ⚙️ 1. Activate the Environment

```bash
conda activate prismpyp
```

---

## 📁 2. Create an Output Directory

```bash
mkdir intersection
```

---

## 🔗 3. Compute the Intersection

Run the following command to take the intersection between real and Fourier domain selections:

```bash
prismpyp intersect \
   --parquet-files output_dir/fft/fft_good_export.parquet output_dir/real/real_good_export.parquet \
   --output-folder intersection \
   --link-type soft \
   --data-path example_data/webp
```

> ⚙️ Use `--link-type soft` to create symbolic links or `--link-type hard` to copy the files instead.

---

## 📦 4. Output Files

The following outputs will be written to the `intersection/` directory:

| File | Description |
|------|--------------|
| `intersection.parquet` | Metadata table containing information for all intersected micrographs |
| `files_in_common.txt` | List of intersected file names |
| Symlinked or copied images | The actual intersected micrographs |

---

This subset represents the most consistent and high-quality micrographs across both domains — ideal for subsequent refinement or reconstruction workflows. 🎯

---

### Next Steps
⬅️ [Back: Interactive Visualization (Phoenix)](phoenix.md)
