## 🔍 Performing dual-domain filtering
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
    --parquet-files output_dir/fft/fft_good_export.parquet output_dir/real/real_good_export.parquet \
    --output-folder intersection \
    --link-type soft \
    --data-path example_data/webp
```

The files in common will be symlinked (if ```--link-type``` is set to ```soft```) or copied (if ```--link-type``` is set to ```hard```) to the ```output_folder``` ```intersection```. The metadata associated with the files in common will be written to ```intersection/intersection.parquet```, and a list of the files in common will be written to ```intersection/files_in_common.txt```.
