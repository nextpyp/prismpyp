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
    
    parser.add_argument('--real-parquet-file', type=str, 
                        help='Path to real parquet file')
    parser.add_argument('--fft-parquet-file', type=str, 
                        help='Path to FFT parquet file')
    
    parser.add_argument('--good-real-classes', nargs='+', type=int,
                        help='Space-separated list of real domain cluster IDs to treat as good (e.g. --good-real-classes 2 5 7)')
    parser.add_argument('--good-fft-classes', nargs='+', type=int,
                        help='Space-separated list of Fourier domain cluster IDs to treat as good (e.g. --good-fft-classes 1 8 9)')
    
    parser.add_argument('--output-folder', type=str, 
                        help='Output folder')
    parser.add_argument('--link-type', type=str, choices=['hard', 'soft'], 
                        help='Link type (hard or soft)')
    parser.add_argument("--webp-path", type=str, 
                        help="Path to where the original .webp files are")
 
    return parser


def filter_df_by_class(df, classes):
    df_filtered = df.copy()
    df_filtered = df[df['cluster_id'].isin(classes)]
    return df_filtered


def rename_micrograph_column(df):
    if 'micrograph_name' in df.columns:
        return df
    elif 'url' in df.columns: # If coming from FFT parquet
        df = df.copy()
        df['micrograph_name'] = (
            df['url']
            .str.split('/').str[-1]
            .str.replace(r'\.mrc\.combined\.jpg$', '', regex=True)
            .str.replace(r'\.combined\.jpg$', '', regex=True)
            .str.replace(r'\.webp$', '', regex=True)
        )
        return df
    elif 'micrograph_id' in df.columns: # If coming from some other type of parquet
        df = df.copy()
        df = df.drop_duplicates(subset='micrograph_id') # One micrograph has many patches, but we don't care about that at this stage
        df['micrograph_name'] = (
            df['micrograph_id'].
            str.split('/').str[-1]
            .str.replace(r'\.webp$', '', regex=True)
            .str.replace(r'\.mrc$', '', regex=True)
        )
        return df
    else:
        raise ValueError(
            f"Parquet must have a 'micrograph_id' or 'url' column. "
            f"Found: {list(df.columns)}"
        )
        
    
def check_is_subset(all_clusters, filtered_clusters):
    if set(filtered_clusters).issubset(set(all_clusters)):
        return True
    else:
        missing = set(filtered_clusters) - set(all_clusters)
        print("Missing the following clusters: ", missing)
        return False

def main(args):
    
    first = args.real_parquet_file
    second = args.fft_parquet_file
    
    good_real_classes = args.good_real_classes
    good_fft_classes = args.good_fft_classes
    filter_by_class = None
    
    if good_real_classes is not None and good_fft_classes is not None:
        filter_by_class = True
        print("Filtering by class ID")
    elif good_real_classes is None and good_fft_classes is not None:
        print("Warning! No good classes in the real domain were specified")
        filter_by_class = True
        good_real_classes = []
    elif good_real_classes is not None and good_fft_classes is None:
        print("Warning! No good classes in the Fourier domain were specified")
        filter_by_class = True
        good_fft_classes = []
    else:
        filter_by_class = False
        print("Filtering by parquet file since classes were not specified")
        
    
    df1 = pandas.read_parquet(first) # Columns: cluster_id, url, prediction_id, timestamp
    df2 = pandas.read_parquet(second) # Columns: cluster_id, url, prediction_id, timestamp
    
    real_clusters = df1['cluster_id'].to_list()
    fft_clusters = df2['cluster_id'].to_list()
    
    if filter_by_class:
        df1_to_filter = filter_df_by_class(df1, good_real_classes)
        df2_to_filter = filter_df_by_class(df2, good_fft_classes)
        real_is_subset = check_is_subset(real_clusters, good_real_classes)
        fft_is_subset = check_is_subset(fft_clusters, good_fft_classes)
        
        if not (real_is_subset and fft_is_subset):
            print("Classes are missing! Please double check")
            return
    else:
        df1_to_filter = df1
        df2_to_filter = df2
        
    df1_to_filter = rename_micrograph_column(df1_to_filter)
    df2_to_filter = rename_micrograph_column(df2_to_filter)
    
    intersection = pandas.merge(df1_to_filter, df2_to_filter, how='inner', on=['micrograph_name'])
    # print("Intersection: ", intersection.head())
    print("Found", len(intersection), "files in common")
        
    # Save the intersection 
    os.makedirs(os.path.join(args.output_folder, "files"), exist_ok=True)
    intersection.to_parquet(os.path.join(args.output_folder, 'intersection.parquet'))
    
    list_of_files = []
    for i, row in intersection.iterrows():
        # Get filename from url
        filename = row['micrograph_name'].split('/')[-1]
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
        
        # Search for filename in webp_path
        if not os.path.exists(os.path.join(args.webp_path, filename)):
            print("File not found: ", filename)
        else:
            new_path = os.path.join(args.output_folder, "files", filename)
            if args.link_type == 'hard':
                os.link(os.path.join(args.webp_path, filename), new_path)
            else:
                if not os.path.exists(new_path):
                    os.symlink(os.path.join(args.webp_path, filename), new_path)
                else:
                    print("Symlink already exists: ", new_path)
    
    with open(os.path.join(args.output_folder, 'files_in_common.txt'), 'w') as f:
        for item in list_of_files:
            f.write("%s\n" % item)
            
if __name__ == "__main__":
    main(add_args().parse_args())