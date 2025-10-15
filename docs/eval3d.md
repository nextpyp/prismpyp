## 🗽 Visualizing your dataset: Creating 3D interactive visualizations
   
   Similar to the previous section, 3D visualization allows us to interact with high-dimensional data in a lower-dimensional plane. In 3D interactive visualization, you can go a step further and manually examine images in the point cloud to determine the composition of regions of the embedding space. You can also select images containing high-quality features and export them for further processing.

   1. If you have already done 2D visualization, you can skip the embedding generation, and simply provide a path to the ```embedding.pth``` file:
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

   2. Otherwise, the embeddings will need to be generated from scratch:
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

   Add the ```--use-fft``` flag according to the domain that your input images are in.

   If this is the first time you have run the ```prismpyp eval3d``` file, you will most likely also want to generate the zipped image thumbnails for visualization. This can be done using the flag ```--zip-images```. It is recommended that you only do this once and reuse the zipped images for future visualizations, unless your micrographs have changed between embeddings, since the process of zipping the images takes a couple of minutes.

   At the end of 3D visualization, the following files will be outputted to ```/path/to/outputs/inference```:
   * ```data_for_export.parquet.zip```: A metadata file containing columns such as ```micrograph_name```, ```embeddings``` (the high-dimensional embedding vector), traditional heuristic metadata like CTF fit, estimated resolution, mean defocus, and ice thickness (if applicable)
   * ```zipped_thumbnail_images.tar.gz```: A zipfile containing a composite image of the micrograph and its corresponding power spectrum.
