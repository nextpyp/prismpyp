# Self-Supervised Learning for Automatic Cryo-EM Micrograph Exploration

This repository implements a SimSiam-based self-supervised pipeline for classifying cryo-EM micrographs based on both real-space and Fourier-space features. The goal is to uncover image quality categories such as vitreous ice, crystalline ice, contaminants, and support film without using manual labels.

## ⚙️ Setting up the environment
Because of the dependency on CUDA and PyTorch-GPU, all of the following instructions should be run from a computer with a GPU. 

This setup process and code has been tested with ```PyTorch/2.4.0``` and  ```cuda/12.1```.

### Install conda
   Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/).

### Clone the repository
   ```bash
   git clone git@github.com:nextpyp/prismpyp.git
   cd prismpyp
   ```
   
### Set up the environment
   
   Using pip:
   ```bash
   conda create -n prismpyp -c conda-forge python=3.12 pip
   conda activate prismpyp
   
   python -m pip install . --extra-index-url https://download.pytorch.org/whl/cu121
   
   conda install -c pytorch -c conda-forge faiss-gpu=1.9.0 # The pip wheel for faiss-gpu does not support python/3.12
   ```

   Or, using Conda (slower, not recommended):
   ```bash
   conda env create -f environment.yml -n prismpyp
   conda activate prismpyp
   ```
  
  Your Conda environment should now be active.

## 🧪 Training the model

### Download the test data
We will use the ```example_data.tar.gz``` file containing micrograph and power spectra images, and metadata from EMPIAR-10379 as an example test case of self-supervised micrograph sorting. The file is available on [Zenodo](https://doi.org/10.5281/zenodo.17161604). 


Select an appropiate location on your file system, create a new folder named ```example_data```, download and unpack the files from zenodo into it:
```bash
mkdir example_data
tar -xvzf example_data.tar.gz -C example_data
```

This command extracts the data into a folder ```example_data``` with the following folders and files:
* ```pkl``` contains the metadata from pre-processing EMPIAR-10379 in nextPYP
* ```webp``` contains ```512 x 512``` images of the micrographs and their corresponding power spectra
* The ```.cs``` file contains metadata from the Manual Curation job in CryoSPARC
* The ```.toml``` file contains microscope data such as the pixel size of the micrographs
* The ```.micrographs``` file contains a list of micrographs that are present in the data, without any file extensions

The data should be structured like this:
```
example_data/
    |
    |- pkl/
    |   |- file1.pkl
    |   |- file2.pkl
    |   |- ...
    |- webp/
    |   |- file1.webp
    |   |- file1_ctffit.webp
    |   |- file2.webp
    |   |- file2_ctffit.webp
    |   |- ...
    |- J7_exposures_accepted_exported.cs
    |- sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs
    |- .pyp_config.toml # This file will only be visible in the directory if you use 'ls -al' to show all files
```

The following instructions assume a directory structure similar to the one above. If your directory structure is different, please note that you may need to change the commands in order to run them properly.

### Build the metadata table
   1. Make the metadata and the training output directory: 
   ```bash
   mkdir -p metadata
   mkdir -p output_dir
   ```

   2. Then, run:
   ```bash
   prismpyp metadata \
    --pkl-path example_data/pkl \
    --output-file metadata/micrograph_table.csv \
    --cryosparc-path example_data/J7_exposures_accepted_exported.cs
   ```

   You can omit ```--cryosparc-path``` if you don’t need relative ice thickness visualization.
  
  This command will produce a ```.csv``` file that populates, for each image in your dataset, the:
  * ```micrograph_name```
  * ```rel_ice_thickness``` (If you provide ```--cryosparc-path```)
  * ```ctf_fit```
  * ```est_resolution```
  * ```avg_motion```
  * ```num_particles```
  * ```mean_defocus```

### Train the SimSiam model
   1. Download the ResNet50 pre-trained weights:
   ```bash
   mkdir -p pretrained_weights
   wget https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar -P pretrained_weights/
   ```

   2. To train the model on real-domain images, run:
   ```bash
   prismpyp train \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/real \
    --metadata-path metadata/micrograph_table.csv \
    -a resnet50 \
    --epochs 100 \
    --batch-size 512 \
    --workers 1 \
    --dim 512 \
    --pred-dim 256 \
    --lr 0.05 \
    --resume pretrained_weights/checkpoint_0099.pth.tar \
    --nextpyp-preproc example_data \
    --multiprocessing-distributed \
    --dist-url 'tcp://localhost:10057' \
    --world-size 1 \
    --rank 0
   ```

   3. For Fourier-domain images, run:
   ```bash
   prismpyp train \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/fft \
    --metadata-path metadata/micrograph_table.csv \
    -a resnet50 \
    --epochs 100 \
    --batch-size 512 \
    --workers 1 \
    --dim 512 \
    --pred-dim 256 \
    --lr 0.05 \
    --resume pretrained_weights/checkpoint_0099.pth.tar \
    --nextpyp-preproc example_data \
    --multiprocessing-distributed \
    --dist-url 'tcp://localhost:10058' \
    --world-size 1 \
    --rank 0 \
    --use-fft
   ```
  
  If you'd rather start from scratch or use your own pretrained model, omit the ```--resume``` flag or point it to your own ```.pth.tar``` checkpoint.
  
  During training, the per-batch loss and collapse level will be written to the terminal. At the end of training, the following will be available at the output path ```/path/to/output_dir``` specified in the command(s) above:

  * Model checkpoints (```.pth.tar```) for the epoch with the lowest loss (```checkpoints/model_best.pth.tar```), the epoch with the lowest collapse (```checkpoints/model_lowest_collapse.pth.tar```), and the last epoch (```checkpoints/model_last.pth.tar```)
  
  * Loss during training (```total_loss.webp```)
    * This plot is good to look at to confirm that training converged. If it has, then the loss line will stabilize. Lower (more negative) numbers are better.
  * Collapse level during training (```collapse_level.webp```)
    * This plot is good to look at to conform that the model has not learned nonsensical representations of image features. Higher numbers are better.
  * The settings for training (```training_config.yaml```)
  
## 🖼️ Visualizing your dataset
### Generate embeddings and 2D visualizations for the entire dataset

   Once the model has finished training, we can generate static 2D visualizations of the feature vectors to examine the different types of micrograph and power spectrum features present in the dataset.

   1. To perform inference on real-domain images:
   ```bash
   prismpyp eval2d \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/real \
    --metadata-path metadata/micrograph_table.csv \
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
    --min-dist-umap 0 \
    --nextpyp-preproc example_data \
   ```

   2. To perform inference on Fourier-domain images:
   ```bash
   prismpyp eval2d \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/fft \
    --metadata-path metadata/micrograph_table.csv \
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
    --nextpyp-preproc example_data \
    --use-fft
   ```

   3. If you have already produced embeddings, you can skip the inference step and simply project the embedding vectors onto 2D by providing a path to the embeddings file, like so:
   ```bash
   prismpyp eval2d \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/fft \
    --metadata-path metadata/micrograph_table.csv \
    --embedding-path output_dir/fft/inference/embeddings.pth \
    -a resnet50 \
    --dist-url "tcp://localhost:10048" \
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
    --nextpyp-preproc example_data
   ```
   Include the ```--use-fft``` flag if you are running on Fourier-domain data.

   Inference will output the following files to a folder ```/path/to/output/<real or fft>/inference```:
   * ```embeddings.pth```: The high-dimensional feature vectors produced during inference for the images in the dataset.
   * ```nearest_neighbors_x.webp```: The 8 nearest neighbors (points in the high-dimensional embedding space with the smallest Euclidean distance) of a randomly-sampled point in the dataset.
   * ```scatter_plot_<method>.webp```: A point cloud scatter plot produced by projecting the high-dimensional data to 2D using either PCA, UMAP, or tSNE.
   * ```thumbnail_plot_<method>_<ps or mg>.webp```: Same as the scatter plot above, but instead of representing each image with a point, we show either the micrograph (```mg```) or the power spectrum (```ps```) as a static 2D preview of the embedding space. Useful for determining general areas or visual patterns in the data.
  
### Creating 3D interactive visualizations
   
   Similar to the section above, 3D visualization allows us to interact with high-dimensional data in a lower-dimensional plane. In 3D interactive visualization, you can go a step further and manually examine images in the point cloud to determine the composition of regions of the embedding space. You can also select images containing high-quality features and export them for further processing.

   1. If you have already done 2D visualization, you can skip the embedding generation, and simply provide a path to the ```embedding.pth``` file:
   ```bash
   prismpyp eval3d \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/real \
    --metadata-path metadata/metadata_table.csv \
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
    --min-dist-umap 0 \
    --nextpyp-preproc example_data
   ``` 

   2. Otherwise, the embeddings will need to be generated from scratch:
   ```bash
   prismpyp eval3d \
    --micrograph-list example_data/sp-preprocessing-fhgRaEnEqUsEFrUj.micrographs \
    --output-path output_dir/real \
    --metadata-path metadata/metadata_table.csv \
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
    --min-dist-umap 0 \
    --nextpyp-preproc example_data
   ```

   Add the ```--use-fft``` flag according to the domain that your input images are in.

   If this is the first time you have run the ```prismpyp eval3d``` file, you will most likely also want to generate the zipped image thumbnails for visualization. This can be done using the flag ```--zip-images```. It is recommended that you only do this once and reuse the zipped images for future visualizations, unless your micrographs have changed between embeddings, since the process of zipping the images takes a couple of minutes.

   At the end of 3D visualization, the following files will be outputted to ```/path/to/outputs/inference```:
   * ```data_for_export.parquet.zip```: A metadata file containing columns such as ```micrograph_name```, ```embeddings``` (the high-dimensional embedding vector), traditional heuristic metadata like CTF fit, estimated resolution, mean defocus, and ice thickness (if applicable)
   * ```zipped_thumbnail_images.tar.gz```: A zipfile containing a composite image of the micrograph and its corresponding power spectrum.

## Visualizing 3D results in Phoenix

We use Phoenix to do interactive visualization and selection of good micrographs. For the best visualization experience, Phoenix should be installed locally (i.e.: not on a remote cluster). As such, the following steps are recommended to be carried out on your local machine.

Install Phoenix from either the ```.yml``` file:
```bash
conda env create -f phoenix.yml -n phoenix
conda activate phoenix
```

Or using pip:
```bash
conda create -n phoenix -c conda-forge python=3.8 pip
conda activate phoenix
python -m pip install -r requirements-phoenix.txt
```

The following instructions use the inference results generated from the real domain inputs.

1. Copy the output files (```output_dir/real/inference/data_for_export.parquet.zip```, ```output_dir/real/inference/zipped_thumbnail_images.tar.gz```) from 3D inference to your local machine (e.g., ```real```).
2. Make a folder for the thumbnail images:
```bash
mkdir -p real/thumbnail_images
```
3. Unzip the thumbnail images file:
```bash
tar -xvzf zipped_thumbnail_images.tar.gz -C real/thumbnail_images
```

4. Start an HTTP server to host the thumbnail images:
```bash
cd real
python -m http.server 5004
```

5. In another Bash window, download the visualization script and start the visualization code:
```bash
wget https://github.com/nextpyp/prismpyp/blob/main/scripts/visualizer.py

python visualizer.py \
    real/data_for_export.parquet.zip \
    --port 5004 \
    --which-embedding umap
```

Make sure that the port used in steps 4 and 5 are the same!

This step will output something like this to the terminal:
```bash
                                          embeddings  
0  [-0.07116878777742386, -0.02092127315700054, 0...  
1  [0.03098396770656109, -0.012207355350255966, 0...  
2  [-0.001951615558937192, -0.059190645813941956,...  
3  [0.04287125542759895, 0.055738892406225204, -0...  
4  [-0.03130991384387016, -0.024252604693174362, ...  
🌍 To view the Phoenix app in your browser, visit http://localhost:54116/
📺 To view the Phoenix app in a notebook, run `px.active_session().view()`
📖 For more information on how to use Phoenix, check out https://docs.arize.com/phoenix
```

You should now be able to access the interactive session by visiting ```http://localhost:54116/``` in a web browser.

The same steps can be followed to do interactive selection on the Fourier domain inputs:
1. Copy the output files (```output_dir/fft/inference/data_for_export.parquet.zip```, ```output_dir/fft/inference/zipped_thumbnail_images.tar.gz```) from 3D inference to your local machine (e.g., ```fft```).
2. Make a folder for the thumbnail images:
```bash
mkdir -p fft/thumbnail_images_dir
```
3. Unzip the thumbnail images file:
```bash
tar -xvzf zipped_thumbnail_images.tar.gz -C fft/thumbnail_images_dir
```

4. In another Bash window, start an HTTP server to host the thumbnail images:
```bash
cd fft
screen
python -m http.server 5004
```

5. In one Bash window, start the visualization code:
```bash
python visualizer.py \
    fft/data_for_export.parquet.zip \
    --port 5004 \
    --which-embedding umap
```

For both domains, the lasso selection output will be saved and downloaded to a ```.parquet``` file, e.g., ```real/real_good_export.parquet``` or ```fft/fft_good_export.parquet```

## Performing dual-domain filtering
After you have selected the high-quality subsets in both the real and Fourier domains, we can take the intersection to find the images that have high-quality features in both domains.

If you wish to run this on a remote cluster, transfer the .parquet files from your local machine to the cluster.

1. Activate the Conda environment
```bash
conda activate prismpyp
```

2. Create a new directory where you will store your new output files:
```bash
mkdir intersection
```

3. Take the intersection of the exported ```.parquet``` files:
```bash
prismpyp intersect \
    --parquet_files output_dir/fft/fft_good_export.parquet output_dir/real/real_good_export.parquet \
    --output_folder intersection \
    --link_type soft \
    --data_dir example_data/webp
```

The files in common will be symlinked (if ```--link_type``` is set to ```soft```) or copied (if ```--link_type``` is set to ```hard```) to the ```output_folder``` ```intersection```. The metadata associated with the files in common will be written to ```intersection/intersection.parquet```, and a list of the files in common will be written to ```intersection/files_in_common.txt```.

## Zenodo Files

The Zenodo link for this project contains the following files:
* ```model_weights.tar.gz```: Trained model weights for the real domain input (```real_model_best.pth.tar```) and the Fourier domain input (```fft_model_best.pth.tar```)
* ```fft_good_export.parquet```: Data points that have high-quality features in the Fourier domain
* ```real_good_export.parquet```: Data points that have high-qualtiy features in the real domain

By taking the intersection of ```fft_good_export.parquet``` and ```real_good_export.parquet```, you can obtain the 862 micrographs that were used to obtain the 2.9A structure in the paper.