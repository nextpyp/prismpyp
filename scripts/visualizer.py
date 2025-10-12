import argparse
import os
import socket

import numpy as np
import pandas as pd
from scipy.stats import zscore
import phoenix as px


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_file", type=str, help="Path to .parquet.zip file containing embeddings")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Phoenix server on")
    parser.add_argument("--which-embedding", type=str, default="umap", choices=["pca", "umap", "tsne"],
                        help="Which embedding to visualize")
    return parser.parse_args()


def check_port_in_use(port, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True


def find_next_available_port(start_port=7000, host='127.0.0.1'):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except socket.error:
                port += 1


def concat_fit_coords(df, which_embedding):
    """Combine separate embedding dimension columns into a single np.array per row."""
    try:
        df[f"{which_embedding}_fit"] = df.apply(
            lambda row: np.array([row[f"{which_embedding}_fit_x"],
                                  row[f"{which_embedding}_fit_y"],
                                  row[f"{which_embedding}_fit_z"]]),
            axis=1
        )
        df.drop(columns=[f"{which_embedding}_fit_x", f"{which_embedding}_fit_y", f"{which_embedding}_fit_z"],
                inplace=True)
    except KeyError:
        df[f"{which_embedding}_fit"] = df.apply(
            lambda row: np.array([row[f"{which_embedding}_fit_x"],
                                  row[f"{which_embedding}_fit_y"]]),
            axis=1
        )
        df.drop(columns=[f"{which_embedding}_fit_x", f"{which_embedding}_fit_y"], inplace=True)
    return df


def main():
    args = parse_args()
    port = args.port
    which_embedding = args.which_embedding

    print(f"Using port {port}")

    parquet_file = args.parquet_file
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"File '{parquet_file}' not found.")

    try:
        data = pd.read_parquet(parquet_file)
    except Exception as e:
        raise Exception(f"Failed to read parquet file '{parquet_file}': {e}")

    # Modify image thumbnail paths to serve locally
    data['url'] = data['image_thumbnails'].apply(
        lambda x: os.path.join(f"http://localhost:{port}/thumbnail_images", os.path.basename(x) + ".combined.jpg")
    )

    print(data.head())

    # Define Phoenix schema
    try:
        schema = px.Schema(
            prediction_label_column_name="cluster_id",
            feature_column_names=[
                'micrograph_name', 'ctf_fit', 'est_resolution',
                'rel_ice_thickness', 'cluster_id', 'mean_defocus'
            ],
            embedding_feature_column_names={
                "image_embedding": px.EmbeddingColumnNames(
                    vector_column_name="embeddings",
                    link_to_data_column_name="url"
                )
            }
        )
    except Exception:
        # Fallback in case the full schema fails (e.g. missing columns)
        schema = px.Schema(
            prediction_label_column_name="cluster_id",
            tag_column_names=['micrograph_name', f"{which_embedding}_fit"],
            embedding_feature_column_names={
                "image_embedding": px.EmbeddingColumnNames(
                    vector_column_name=f"{which_embedding}_fit",
                    link_to_data_column_name="url"
                )
            }
        )

    dataset = px.Dataset(dataframe=data, schema=schema)
    session = px.launch_app(dataset)


if __name__ == "__main__":
    main()