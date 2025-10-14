import argparse
import glob
import re
import shutil
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm


def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description='Parse .pkl files to retrieve CTF fit/estimated resolution/average drift/number of particles metadata')
    else:
        # this script is called from prismpyp.__main__ entry point, in which case a parser is already created
        pass
    
    parser.add_argument('--pkl-path', type=str, 
                        help='Path to the pkl files')
    parser.add_argument('--cryosparc-path', type=str, default=None, 
                        help='Path to CryoSPARC Curate Exposure job exported .cs file (optional)')
    parser.add_argument('--output-dir', type=str, help='Output file name')
    parser.add_argument('--micrographs-list', type=str, default=None,
                        help='Optional path to a text file containing a list of micrograph names (one per line) to filter the output')

    return parser

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Parse .pkl files to retrieve CTF fit/estimated resolution/average drift/number of particles metadata')
    parser.add_argument('pkl_path', type=str, 
                        help='Path to the pkl files')
    parser.add_argument('--cryosparc_path', type=str, default=None, 
                        help='Path to CryoSPARC Curate Exposure job exported .cs file (optional)')
    parser.add_argument('output_file', type=str, help='Output file name')
    return parser.parse_args()
"""
    
def calculate_avg_motion(pkl):
    avg_dx = np.mean(np.abs(pkl['drift']['dx']))
    avg_dy = np.mean(np.abs(pkl['drift']['dy']))
    avg_drift = np.sqrt(avg_dx**2 + avg_dy**2)
    
    return avg_drift

def build_table(path_to_pkls, path_to_cs_file, output_dir):
    if path_to_cs_file is not None:
        curate_exposures = np.load(path_to_cs_file)
        # Retrieve ice thickness entry --> key: 'ctf_stats/ice_thickness_rel'
        micrograph_name = []
        for filename in curate_exposures['micrograph_blob/path']:
            try:
                decoded_filename = filename.decode('utf-8')
            except AttributeError:
                decoded_filename = filename
            name = os.path.basename(decoded_filename).split('.')[0]
            
            # Regular expression pattern to match a sequence of digits followed by '_'
            pattern = r'\d+_'  
            split_result = re.split(pattern, name, maxsplit=1)
            if len(split_result) > 1:
                result = split_result[1]
            else:
                result = name  # In case no match is found, return the original string
            # print(name)
            micrograph_name.append(result)
            
        ice_thickness = curate_exposures['ctf_stats/ice_thickness_rel']
        ice_thickness_dict = dict(zip(micrograph_name, ice_thickness))
        
        # Retrieve pixel size and write to file
        if "micrograph_blob/psize_A" in curate_exposures.dtype.names:
            pixel_size = curate_exposures["micrograph_blob/psize_A"][0]
        elif "micrograph_blob_non_dw/psize_A" in curate_exposures.dtype.names:
            pixel_size = curate_exposures["micrograph_blob_non_dw/psize_A"][0]
        else:
            raise KeyError("Could not find pixel size column in the .cs file")
        pixel_size_file = os.path.join(output_dir, 'pixel_size.txt')
        with open(pixel_size_file, 'w') as f:
            f.write(str(pixel_size) + '\n')
    
    pkl_files = os.listdir(path_to_pkls)
    dbase = {}
    for pkl_file in tqdm(pkl_files, desc="Processing pkl files"):
        path_to_pkl_file = os.path.join(path_to_pkls, pkl_file)
        filename = pkl_file.split('.')[0]
        filename = os.path.basename(pkl_file).split('.')[0]
        try:
            pkl = pd.read_pickle(path_to_pkl_file)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
        ctf_fit = pkl['ctf'].loc['cc'].values[0]
        est_resolution = pkl['ctf'].loc['cccc'].values[0]
        avg_motion = calculate_avg_motion(pkl)
        num_particles = len(pkl['box']) if 'box' in pkl.keys() else 0
        df1 = pkl['ctf'].loc['mean_df'].values[0]
        
        dbase[filename] = {
            'rel_ice_thickness' : ice_thickness_dict[filename] if path_to_cs_file is not None else 1,
            'ctf_fit' : ctf_fit,
            'est_resolution' : est_resolution,
            'avg_motion' : avg_motion,
            'num_particles' : num_particles,
            'mean_defocus' : df1,
        }
    
    dbase_df = pd.DataFrame.from_dict(dbase, orient='index')
    dbase_df.reset_index(inplace=True)
    dbase_df.rename(columns={'index': 'micrograph_name'}, inplace=True)
    output_file = os.path.join(output_dir, 'micrograph_metadata.csv')
    if os.path.exists(output_file):
        print("Warning! Output file {} already exists. Overwriting...".format(output_file))
    dbase_df.to_csv(output_file, index=False)
    
    # Search preprocessing dir for .micrographs file and copy to metadata dir under 'all_micrographs_list.micrographs'
    preprocessing_path = os.path.dirname(path_to_pkls)
    micrographs_list_file = glob.glob(os.path.join(preprocessing_path, '*.micrographs'))
    if len(micrographs_list_file) > 0:
        micrographs_list_file = micrographs_list_file[0]
        dest_file = os.path.join(output_dir, 'all_micrographs_list.micrographs')
        shutil.copyfile(micrographs_list_file, dest_file)
        print(f"Copied {micrographs_list_file} to {dest_file}")
    
    return

def main(args):
    input_path = args.pkl_path
    csparc_path = args.cryosparc_path
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    build_table(input_path, csparc_path, output_dir)
    return
    
if __name__ == '__main__':
    main(add_args().parse_args())
        