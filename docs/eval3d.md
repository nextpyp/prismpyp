# 3D Interactive Embedding Visualization

3D visualization extends the 2D projection step by allowing **interactive exploration** of your dataset’s embedding space.  
Here, you can manually inspect micrographs within clusters, identify distinct regions (e.g., vitreous vs. crystalline ice), and export high-quality subsets for further analysis.

> Use this step to interpret learned representations and select data subsets for refinement or reconstruction.

---

## 1. Visualize Precomputed Embeddings

If you already generated 2D embeddings, you can skip re-computation and directly visualize them in 3D:

```bash
prismpyp eval3d \
   --output-path output_dir/real \
   --metadata-path metadata_from_nextpyp \
   --embedding-path output_dir/real/inference/embeddings.pth \
   -a resnet50 \
   --dist-url 'tcp://localhost:10038' \
   --world-size 1 \
   --rank 0 \
   --batch-size 512 \
   --workers 1 \
   --gpu 0 \
   --fix-pred-lr \
   --feature-extractor-weights output_dir/real/checkpoints/model_best.pth.tar \
   --evaluate \
   --dim 512 \
   --pred-dim 256 \
   --n-clusters 10 \
   --num-neighbors 10 \
   --min-dist-umap 0
```

> Reuse embeddings from the `eval2d` step to speed up processing and maintain consistency across visualization scales.

---

## 2. Generate New 3D Embeddings from Scratch

If you haven’t yet produced embeddings, you can create them during the 3D visualization process:

```bash
prismpyp eval3d \
   --output-path output_dir/real \
   --metadata-path metadata_from_nextpyp \
   -a resnet50 \
   --dist-url 'tcp://localhost:10028' \
   --world-size 1 \
   --rank 0 \
   --batch-size 512 \
   --workers 1 \
   --gpu 0 \
   --fix-pred-lr \
   --feature-extractor-weights output_dir/real/checkpoints/model_best.pth.tar \
   --evaluate \
   --dim 512 \
   --pred-dim 256 \
   --n-clusters 10 \
   --num-neighbors 10 \
   --min-dist-umap 0
```

Add the `--use-fft` flag to process Fourier-domain data.

---

## 3. Generate Thumbnails for Visualization

If this is your first time running `prismpyp eval3d`, you may want to generate **zipped image thumbnails** for interactive rendering:

```bash
prismpyp eval3d \
   --output-path output_dir/real \
   --metadata-path metadata_from_nextpyp \
   --embedding-path output_dir/real/inference/embeddings.pth \
   -a resnet50 \
   --feature-extractor-weights output_dir/real/checkpoints/model_best.pth.tar \
   --zip-images
```

> ⚠️ The zipping process may take a few minutes. Only rerun with `--zip-images` if your micrographs have changed since the last embedding.

---

## 4. Output Files

At the end of visualization, the following files will be created in `/path/to/outputs/inference`:

| File | Description |
|------|--------------|
| `data_for_export.parquet.zip` | Metadata table with micrograph names, embeddings, and CTF/defocus/ice thickness metrics |
| `zipped_thumbnail_images.tar.gz` | Composite thumbnails combining micrograph and power spectrum for interactive visualization |

---

Your 3D embeddings are now ready for interactive inspection in **Phoenix** — the visualization tool described in the next section.

---

### Next Steps
⬅️ [Back: 2D Embedding Generation](eval2d.md) | ➡️ [Next: Interactive Visualization (Phoenix)](phoenix.md)
