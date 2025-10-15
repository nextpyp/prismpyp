import os
import pandas
import argparse

def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description='Get intersection of mrc files')
 
    else:
        # this script is called from prismpyp.__main__ entry point, in which case a parser is already created
        pass
    
    parser.add_argument('--parquet-files', nargs='+', 
                        help='List of parquet files')
    parser.add_argument('--output-folder', type=str, 
                        help='Output folder')
    parser.add_argument('--link-type', type=str, choices=['hard', 'soft'], 
                        help='Link type (hard or soft)')
    parser.add_argument("--data-path", type=str, 
                        help="Path to where the original .mrc files are")
 
    return parser

def main(args):

    parquet_files = args.parquet_files
    first = True
    intersection = None
    
    assert len(parquet_files) == 2, "Expecting exactly 2 .parquet files, instead got " + str(len(parquet_files))
    first = parquet_files[0]
    second = parquet_files[1]
    
    df1 = pandas.read_parquet(first) # Columns: cluster_id, url, prediction_id, timestamp
    df2 = pandas.read_parquet(second) # Columns: cluster_id, url, prediction_id, timestamp
    # print(df1.head(), df2.head())
    
    intersection = pandas.merge(df1, df2, how='inner', on=['url'])
    # print("Intersection: ", intersection.head())
    print("Found", len(intersection), "files in common")
        
    # Save the intersection 
    os.makedirs(os.path.join(args.output_folder, "files"), exist_ok=True)
    intersection.to_parquet(os.path.join(args.output_folder, 'intersection.parquet'))
    
    list_of_files = []
    for i, row in intersection.iterrows():
        # Get filename from url
        filename = row['url'].split('/')[-1]
        # Handle edge cases for filename
        if filename.endswith(".mrc.combined.jpg"):
            filename = filename.replace(".mrc.combined.jpg", ".webp")
        elif filename.endswith(".combined.jpg"):
            filename = filename.replace(".combined.jpg", ".webp")
        
        if filename.endswith('.webp'):
            filename = filename
        else:
            filename = filename + '.webp'
        
        list_of_files.append(filename)
        
        # Search for filename in data_path
        if not os.path.exists(os.path.join(args.data_path, filename)):
            print("File not found: ", filename)
        else:
            new_path = os.path.join(args.output_folder, "files", filename)
            if args.link_type == 'hard':
                os.link(os.path.join(args.data_path, filename), new_path)
            else:
                os.symlink(os.path.join(args.data_path, filename), new_path)
    
    with open(os.path.join(args.output_folder, 'files_in_common.txt'), 'w') as f:
        for item in list_of_files:
            f.write("%s\n" % item)
            
if __name__ == "__main__":
    main(add_args().parse_args())