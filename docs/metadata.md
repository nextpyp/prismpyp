## Build the metadata table

   1. Make the training output directory: 
   ```bash
   mkdir -p output_dir
   ```
### Using pre-processing outputs from nextPYP

   1. Make the metadata output directory:
   ```bash
   mkdir -p metadata_from_nextpyp
   ```
   2. To build the metadata using information from nextPYP's preprocessing, run:
   ```bash
   prismpyp metadata_nextpyp \
    --pkl-path example_data/pkl \
    --output-dir metadata_from_nextpyp \
    --cryosparc-path example_data/J7_exposures_accepted_exported.cs
   ```

   You can omit ```--cryosparc-path``` if you don’t need relative ice thickness visualization.
  
### Using cryoSPARC outputs

We build the metadata from cryoSPARC by using the outputs of the Patch CTF Estimation and CTFFIND4 jobs. Since these jobs both take aligned micrographs as inputs, you may also need to perform frame averaging and run Motion Correction to convert from raw frames to aligned micrographs. For the test dataset, the data is deposited in EMPIAR as aligned frames, so we skip the frame averaging and motion correction steps.
  1. Export the outputs of the ```Import```, ```Patch CTF Estimation```, and ```CTFFIND4``` jobs. Note the location of the resulting ```.cs``` file.
     1. For the sake of this example, let's assume:
        1. The ```Import Micrographs``` job is ```J1```
        2. ```Patch CTF Estimation``` is ```J2```
        3. ```CTFFIND``` is ```J3```
        4. The cryoSPARC project directory is located at the absolute path ```/cryosparc/output/dir```.
  2. Make the metadata output directory:
   ```bash
   mkdir -p metadata_from_cryosparc
   ```
  3. Build the metadata from cryoSPARC outputs:
   ```bash
   prismpyp metadata_cryosparc \
      --imported-dir "/cryosparc/output/dir/J1/imported" \
      --patch-ctf-file "/cryosparc/output/dir/J2/J2_passthrough_exposures_accepted.cs" \
      --ctffind-dir "/cryosparc/output/dir/J3/ctffind_output" \
      --ctffind-file "/cryosparc/output/dir/exports/groups/J3_exposures_success/J3_exposures_success_exported.cs" \
      --output-dir metadata_from_cryosparc
   ```

### Code outputs
Both commands will produce a ```micrograph_metadata.csv``` file that populates, for each image in your dataset, the:
   * ```micrograph_name```
   * ```rel_ice_thickness``` (If you provide ```--cryosparc-path```)
   * ```ctf_fit```
   * ```est_resolution```
   * ```avg_motion```
   * ```num_particles```
   * ```mean_defocus```

In addition, it will also produce the following files:
* ```pixel_size.txt```: A text file containing the microscope's pixel size for this experiment.
* ```all_micrographs_list.micrographs```: A list of all micrographs, without the file extension
* ```webp```: A directory containing ```.webp``` image files for the micrographs and power spectra estimated from CTFFIND4.

For the subsequent commands, we will use ```metadata_from_nextpyp``` as the metadata location. But, you can easily specify another location by changing ```--metadata-path``` to either ```metadata_from_nextpyp``` or ```metadata_from_cryosparc```, depending on which software you used to process your images.
