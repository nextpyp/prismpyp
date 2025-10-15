## 🖼️ Visualizing your dataset: Generate embeddings and 2D visualizations for the entire dataset

   Once the model has finished training, we can generate static 2D visualizations of the feature vectors to examine the different types of micrograph and power spectrum features present in the dataset.

   1. To perform inference on real-domain images:
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

   2. To perform inference on Fourier-domain images:
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

   3. If you have already produced embeddings, you can skip the inference step and simply project the embedding vectors onto 2D by providing a path to the embeddings file, like so:
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
   Include the ```--use-fft``` flag if you are running on Fourier-domain data.

   Inference will output the following files to a folder ```/path/to/output/<real or fft>/inference```:
   * ```embeddings.pth```: The high-dimensional feature vectors produced during inference for the images in the dataset.
   * ```nearest_neighbors_x.webp```: The 8 nearest neighbors (points in the high-dimensional embedding space with the smallest Euclidean distance) of a randomly-sampled point in the dataset.
   * ```scatter_plot_<method>.webp```: A point cloud scatter plot produced by projecting the high-dimensional data to 2D using either PCA, UMAP, or tSNE.
   * ```thumbnail_plot_<method>_<ps or mg>.webp```: Same as the scatter plot above, but instead of representing each image with a point, we show either the micrograph (```mg```) or the power spectrum (```ps```) as a static 2D preview of the embedding space. Useful for determining general areas or visual patterns in the data.