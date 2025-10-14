import argparse
import re
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import math
import glob

from prismpyp.utils.csparc_file_viewer import CSparcFileViewer
from prismpyp.utils.imageio import load_mrc, mrc2png

def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description='Parse .pkl files to retrieve CTF fit/estimated resolution/average drift/number of particles metadata')
    else:
        # this script is called from prismpyp.__main__ entry point, in which case a parser is already created
        pass
    
    parser.add_argument('--patch-ctf-file', type=str, 
                        help='Path to CryoSPARC Patch CTF Estimation job exported .cs file')
    parser.add_argument('--ctffind-file', type=str, 
                        help='Path to CryoSPARC CTF Estimation job exported .cs file')
    parser.add_argument('--ctffind-dir', type=str,
                        help='Path to directory containing ctffind output .mrc files')
    parser.add_argument('--imported-dir', type=str,
                        help='Path to directory containing imported micrographs')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files')

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

def get_pixel_size(viewer):
    if "micrograph_blob/psize_A" in viewer.columns:
        pixel_size = viewer["micrograph_blob/psize_A"][0]
    elif "micrograph_blob_non_dw/psize_A" in viewer.columns:
        pixel_size = viewer["micrograph_blob_non_dw/psize_A"][0]
    else:
        raise KeyError("Could not find pixel size column in the .cs file")
    return pixel_size


def get_clean_file_name(filename, all_filenames):
    basename = os.path.basename(filename)
    # print("basename: {}".format(basename))
    
    for name in all_filenames:
        # name = os.path.basename(name)[:-4]  # remove .mrc extension
        if name in basename:
            # print("Found matching name")
            return name
        
def save_mrcs_to_webp(files, all_mg_list, out_dir, is_ctffind=False, contrast_stretch=False):
    for file in tqdm(files, desc="Converting .mrc files to .webp images"):
        if not file.endswith('.mrc'):
            print(f"Skipping non-.mrc file: {file}")
            continue
        
        try:
            data = load_mrc(file)
            if data.ndim == 3:
                if data.shape[0] == 1:
                    data = data[0]
                else:
                    print(f"Warning: {file} has more than one slice ({data.shape[0]}). Using the first slice.")
                    data = data[0]
            
            # print("data shape: {}".format(data.shape))
            this_file_name = get_clean_file_name(file, all_mg_list)
            if is_ctffind:
                if not this_file_name.endswith('_ctffind'):
                    this_file_name += "_ctffind"
            # print(f"file: {file}")
            # print(f"this_file_name: {this_file_name}")
            
            if os.path.splitext(this_file_name)[1] == '':
                out_img_file = os.path.join(out_dir, this_file_name + '.webp')
            elif not this_file_name.endswith('.mrc'):
                out_img_file = os.path.join(out_dir, this_file_name.replace('.mrc', '.webp'))
            # print("Saving image to: {}".format(out_img_file))
            mrc2png(data, output_dims=(512, 512), outname=out_img_file, contrast_stretch=contrast_stretch)
        except Exception as e:
            print(f"Could not process file: {file}: {e}")
            # print("Done!")
            # while True:
            #     count = 0
            #     count += 1

def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("Output files will be written to directory: {}".format(output_dir))
    patch_ctf_file = args.patch_ctf_file
    ctffind_file = args.ctffind_file
    
    # Read .cs files
    patch_ctf_viewer = CSparcFileViewer(patch_ctf_file)
    patch_ctf_viewer_no_ext = CSparcFileViewer(patch_ctf_file)
    ctffind_viewer = CSparcFileViewer(ctffind_file)
    ctffind_viewer_no_ext = CSparcFileViewer(ctffind_file)
    
    # Insert cleaned up micrographs column
    patch_ctf_viewer_no_ext.add_trimmed_path_column(keep_extension=False)
    ctffind_viewer_no_ext.add_trimmed_path_column(keep_extension=False)
    patch_ctf_viewer.add_trimmed_path_column()
    ctffind_viewer.add_trimmed_path_column()
    
    # Convert to dataframes
    patch_ctf_df = patch_ctf_viewer.to_dataframe()
    ctffind_df = ctffind_viewer.to_dataframe()
    
    # Write pixel_size to file
    pixel_size = get_pixel_size(patch_ctf_viewer)
    with open(os.path.join(output_dir, "pixel_size.txt"), 'w') as f:
        f.write(str(pixel_size) + '\n')
        
    # Write micrographs list to file
    patch_mg_list = patch_ctf_viewer_no_ext.trimmed_path
    ctffind_mg_list = ctffind_viewer_no_ext.trimmed_path
    all_mg_list = sorted(set(patch_mg_list).union(set(ctffind_mg_list)))
    mg_list_file = os.path.join(output_dir, "all_micrographs_list.micrographs")
    with open(mg_list_file, 'w') as f:
        for mg in all_mg_list:
            f.write(mg + '\n')
    
    # Collate CTF info
    all_ctf_info = pd.merge(patch_ctf_df, ctffind_df, how='outer', left_on='trimmed_path', right_on='trimmed_path', suffixes=('_patch', '_ctffind'))
    
    print(all_ctf_info.columns)
    dbase_df = all_ctf_info[[
        'ctf_stats/ice_thickness_rel', 
        'ctf/cross_corr_ctffind4_ctffind', 
        'ctf/ctf_fit_to_A_ctffind', 
        'ctf/df1_A_ctffind'
    ]]

    dbase_df.rename(columns={
        'ctf_stats/ice_thickness_rel': 'rel_ice_thickness',
        'ctf/cross_corr_ctffind4_ctffind': 'ctf_fit',
        'ctf/ctf_fit_to_A_ctffind': 'est_resolution',
        'ctf/df1_A_ctffind': 'mean_defocus'
    }, inplace=True)
    
    print("[DEBUGGING] len(dbase_df['rel_ice_thickness']): {}".format(len(dbase_df['rel_ice_thickness'])))
    print("[DEBUGGING] len(dbase_df['ctf_fit']): {}".format(len(dbase_df['ctf_fit'])))
    print("[DEBUGGING] len(dbase_df['est_resolution']): {}".format(len(dbase_df['est_resolution'])))
    print("[DEBUGGING] len(dbase_df['mean_defocus']): {}".format(len(dbase_df['mean_defocus'])))
    print(len(patch_ctf_viewer.trimmed_path))
    dbase_df['micrograph_name'] = patch_ctf_viewer.trimmed_path
    dbase_df['num_particles'] = 0
    dbase_df['avg_motion'] = 0.0
    dbase_df.reset_index(drop=True, inplace=True)

    metadata_file = os.path.join(output_dir, "micrograph_metadata.csv")
    if os.path.exists(metadata_file):
        print("Warning! Output file {} already exists. Overwriting...".format(metadata_file))
    dbase_df.to_csv(metadata_file, index=False)
    
            
    # Write CTFFIND .mrc files to .webp images
    if args.ctffind_dir is not None:
        os.makedirs(os.path.join(output_dir, "webp"), exist_ok=True)
        ctffind_mrc_dir = args.ctffind_dir
        ctffind_mrc_files = glob.glob(os.path.join(ctffind_mrc_dir, "*_ctffit.mrc"))
        save_mrcs_to_webp(ctffind_mrc_files, all_mg_list, os.path.join(output_dir, "webp"), is_ctffind=True)
    
    # Write imported micrographs to .webp images
    if args.imported_dir is not None:
        os.makedirs(os.path.join(output_dir, "webp"), exist_ok=True)
        imported_mrc_dir = args.imported_dir
        imported_mrc_files = glob.glob(os.path.join(imported_mrc_dir, "*.mrc"))
        save_mrcs_to_webp(imported_mrc_files, all_mg_list, os.path.join(output_dir, "webp"), contrast_stretch=True)
    
    return

    
if __name__ == '__main__':
    main(add_args().parse_args())
        