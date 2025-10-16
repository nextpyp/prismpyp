# 2D Embedding Generation and Visualization

Once the model has finished training, you can generate **static 2D visualizations** of learned feature vectors.  
These embeddings reveal structural patterns across your dataset — highlighting variations in ice type, contamination, or support film quality.

> 💡 The following steps can be applied to both **real-domain** and **Fourier-domain** models.

---

## 1. Generate 2D Embeddings for Real-Domain Images

Perform inference on the trained **real-domain** model:
   ```bash
   prismpyp eval2d \
    --output-path output_dir/real \
    --metadata-path metadata_from_nextpyp \
    -a resnet50 \
    --dist-url "tcp://localhost:10059" \
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
> Use the same model architecture and feature dimensions as in training.  
> The embeddings will be saved to `output_dir/real/inference`.

---

## 2. Generate 2D Embeddings for Fourier-Domain Images

For Fourier-domain embeddings, include the `--use-fft` flag:

   ```bash
   prismpyp eval2d \
    --output-path output_dir/fft \
    --metadata-path metadata_from_nextpyp \
    -a resnet50 \
    --dist-url "tcp://localhost:10050" \
    --world-size 1 \
    --rank 0 \
    --batch-size 512 \
    --workers 1 \
    --gpu 0 \
    --fix-pred-lr \
    --feature-extractor-weights output_dir/fft/checkpoints/model_best.pth.tar \
    --evaluate \
    --dim 512 \
    --pred-dim 256 \
    --n-clusters 10 \
    --num-neighbors 10 \
    --min-dist-umap 0 \
    --use-fft
   ```
> This produces a 2D projection of the learned Fourier-domain embeddings, highlighting frequency-based variation across micrographs.

---

## 3. Project Precomputed Embeddings

If you have already generated embeddings, you can skip the inference step and directly project them to 2D:
   ```bash
   prismpyp eval2d \
    --output-path output_dir/real \
    --metadata-path metadata_from_nextpyp \
    --embedding-path output_dir/real/inference/embeddings.pth \
    -a resnet50 \
    --dist-url "tcp://localhost:10048" \
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
Include the `--use-fft` flag if projecting **Fourier-domain** embeddings.

---

## 4. Output Files

Each inference run creates an `inference/` directory inside your chosen output path, containing:

| File | Description |
|------|--------------|
| `embeddings.pth` | High-dimensional feature vectors for all images |
| `nearest_neighbors_x.webp` | Visualization of 8 nearest neighbors to a random point in embedding space |
| `scatter_plot_<method>.webp` | 2D scatter plot (PCA, UMAP, or t-SNE projection) |
| `thumbnail_plot_<method>_<ps or mg>.webp` | Same as above, but displays micrograph (`mg`) or power spectrum (`ps`) thumbnails instead of points |

These visualizations are useful for spotting image-quality clusters or distinct artifact types.

---

Your 2D embeddings are now ready for exploration and validation!
Next, you’ll learn how to generate **3D embeddings** for higher-dimensional visualization.

---

### Next Steps
⬅️ [Back: Label-Free Feature Learning](train.md) | ➡️ [Next: 3D Embedding Generation](eval3d.md)