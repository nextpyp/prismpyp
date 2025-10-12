import os
import pandas
import argparse
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Get list from Phoenix')
    parser.add_argument('--cs_files', nargs='+', 
                        help='List of parquet files')
    parser.add_argument('--output_folder', type=str, 
                        help='Output folder')
    parser.add_argument('--link_type', type=str, choices=['hard', 'soft'], 
                        help='Link type (hard or soft)')
    parser.add_argument("--data_dir", type=str, 
                        help="Path to where the original .mrc files are")
    
    return parser.parse_args()

def get_filenames(arr):
    ret = []
    for file in arr:
        string_to_search = file.decode('utf-8').split('/')[-1]
        match = re.search(r'^[^_]+_(.*)', string_to_search)
        if match:
            ret.append(match.group(1))
        else:
            print("No match found for: ", string_to_search)
    return ret
    
def main():
    args = parse_args()
    cs_files = args.cs_files
    
    for file in cs_files:
        print("Reading file: ", file)
        basename = os.path.basename(file).split(".")[0] # Use the basename (ex: good_export, bad_export) as the folder to organize the files
        print("Basename: ", basename)
        os.makedirs(os.path.join(args.output_folder, basename), exist_ok=True)
        
        df = np.load(file) # Column of interest: 'micrograph_blob/path'
        filenames = df['micrograph_blob/path']
        cleaned_filenames = get_filenames(filenames)

        for i, filename in enumerate(cleaned_filenames):

            if filename.endswith('.mrc'):
                filename = filename
            else:
                filename = filename + '.mrc'
            
            # Search for filename in data_dir
            if not os.path.exists(os.path.join(args.data_dir, filename)):
                print("File not found: ", filename)
            else:
                new_path = os.path.join(args.output_folder, basename, filename)
                if os.path.exists(new_path):
                    os.remove(new_path)
                
                if args.link_type == 'hard':
                    os.link(os.path.join(args.data_dir, filename), new_path)
                else:
                    os.symlink(os.path.join(args.data_dir, filename), new_path)
            
if __name__ == "__main__":
    main()