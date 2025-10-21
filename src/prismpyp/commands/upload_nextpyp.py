import os
import argparse
from tqdm import tqdm

def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description='Generate good subset of .mrc files for re-uploading to nextPYP')
 
    else:
        # this script is called from prismpyp.__main__ entry point, in which case a parser is already created
        pass
    parser.add_argument('--intersection-folder', type=str, 
                        help='Intersection folder')
    parser.add_argument('--link-type', type=str, choices=['hard', 'soft'], 
                        help='Link type (hard or soft)')
    parser.add_argument("--mrc-path", type=str, 
                        help="Path to where the original .mrc files are")
 
    return parser

def main(args):
    
    if args.mrc_path is not None:
        new_mrc_dir = os.path.join(args.intersection_folder, "mrcs")
        os.makedirs(new_mrc_dir, exist_ok=True)
        
        with open(os.path.join(args.intersection_folder, 'files_in_common.txt'), 'r') as f:
            list_of_files = f.readlines()
            
        for item in tqdm(list_of_files, desc="Linking MRC files"):
            mrc_filename = item.replace('.webp', '.mrc')
            mrc_path = os.path.join(args.mrc_path, mrc_filename)
            new_path = os.path.join(args.output_folder, "mrcs", mrc_filename)
            if os.path.exists(mrc_path):
                if args.link_type == 'hard':
                    os.link(os.path.join(args.mrc_path, mrc_path), new_path)
                else:
                    os.symlink(os.path.join(args.mrc_path, mrc_path), new_path)
            else:
                print("MRC file not found: ", mrc_path)
            
if __name__ == "__main__":
    main(add_args().parse_args())


