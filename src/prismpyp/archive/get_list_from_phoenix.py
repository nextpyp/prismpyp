import os
import pandas
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Get list from Phoenix')
    parser.add_argument('--parquet_files', nargs='+', 
                        help='List of parquet files')
    parser.add_argument('--output_folder', type=str, 
                        help='Output folder')
    parser.add_argument('--link_type', type=str, choices=['hard', 'soft'], 
                        help='Link type (hard or soft)')
    parser.add_argument("--data_dir", type=str, 
                        help="Path to where the original .mrc files are")
    
    return parser.parse_args()

def main():
    args = parse_args()
    parquet_files = args.parquet_files
    
    for file in parquet_files:
        print("Reading file: ", file)
        basename = os.path.basename(file).split(".")[0] # Use the basename (ex: good_export, bad_export) as the folder to organize the files
        
        os.makedirs(os.path.join(args.output_folder, basename), exist_ok=True)
        
        df = pandas.read_parquet(file) # Columns: cluster_id, url, prediction_id, timestamp
        
        # Get filename from url
        filenames = df['url'].apply(lambda x: x.split('/')[-1])
        for i, filename in enumerate(filenames):
            # Filename is of format: xyz.mrc.combined.jpg
            # Trim off the .combined.jpg part
            filename = filename[:-13]
            
            if filename.endswith('.mrc'):
                filenames[i] = filename
            else:
                filenames[i] = filename + '.mrc'
            
            # Search for filename in data_dir
            if not os.path.exists(os.path.join(args.data_dir, filenames[i])):
                print("File not found: ", filenames[i])
            else:
                new_path = os.path.join(args.output_folder, basename, filenames[i])
                if args.link_type == 'hard':
                    os.link(os.path.join(args.data_dir, filenames[i]), new_path)
                else:
                    os.symlink(os.path.join(args.data_dir, filenames[i]), new_path)
            
if __name__ == "__main__":
    main()