import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler

class WeaklySupevisedLabels(object):
    def __init__(self, metadata_path, output_path, min_dist_umap=0.5):
        self.metadata = pd.read_csv(metadata_path)
        self.has_ice_thickness = 'rel_ice_thickness' in self.metadata.columns
        self.output_path = output_path
        
        if self.has_ice_thickness:
            self.ctf_res = self.metadata[['ctf_fit', 'est_resolution', 'rel_ice_thickness']]
        else:
            self.ctf_res = self.metadata[['ctf_fit', 'est_resolution']]
        scaler = StandardScaler()
        self.ctf_res_scaled = pd.DataFrame(scaler.fit_transform(self.ctf_res), columns=self.ctf_res.columns)
        
        nearest_hundred = round(len(self.metadata['micrograph_name']) * 0.1/100) * 100
        nearest_tens = round(len(self.metadata['micrograph_name']) * 0.15)
        self.num_neighbors = nearest_tens
        # print("[DEBUGGING] Number of neighbors: ", self.num_neighbors, " for ", len(self.metadata['micrograph_name']), " micrographs")
        self.min_dist_umap = min_dist_umap
        
        self.embedding_2d = None
        self.cluster_labels = None
        self.num_clusters = None
    
    def run(self):
        self.umap_projection()
        self.hdbscan_clustering()
    
    def __getitem__(self, idx):
        if self.filtered_labels is None:
            self.hdbscan_clustering()
        return self.filtered_labels[idx]
        
    def umap_projection(self):
        reducer = umap.UMAP(n_neighbors=self.num_neighbors, random_state=42, min_dist=self.min_dist_umap)
        self.embedding_2d = reducer.fit_transform(self.ctf_res_scaled)
        self.plot_projections(
            'UMAP projection with n_neighbors = {}'.format(self.num_neighbors),
            'umap_projection_n_neighbors={}'.format(self.num_neighbors),
        )
    
    def hdbscan_clustering(self):
        clusterer = HDBSCAN(min_cluster_size=self.num_neighbors//10)
        self.cluster_labels = clusterer.fit_predict(self.ctf_res_scaled)
        self.num_clusters = len(np.unique(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        print(f"Number of clusters: {len(np.unique(self.cluster_labels))}")
        print(f"Number of clusters (excluding noise): {self.num_clusters}")
        
        if self.embedding_2d is None:
            self.umap_projection()
        
        self.plot_hdbscan_clusters(clusterer, 
                                   self.embedding_2d, 
                                   self.cluster_labels,
                                   title=f"Estimated number of clusters: {self.num_clusters}",
                                   save_title='hdbscan_clusters')
        
        # Get rid of noisy labels
        self.valid_indices = self.cluster_labels >= 0
        self.filtered_labels = self.cluster_labels
    
    def get_labels(self):
        if self.filtered_labels is None:
            self.hdbscan_clustering()
        return self.filtered_labels
    
    def get_indices(self):
        if self.valid_indices is None:
            self.hdbscan_clustering()
        return self.valid_indices

    def plot_projections(self, title, save_title, labels=None):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=self.embedding_2d[:, 0], y=self.embedding_2d[:, 1], s=5)
        if labels is not None:
            sns.scatterplot(x=self.embedding_2d[:, 0], y=self.embedding_2d[:, 1], hue=labels, s=5, palette='viridis', legend='full')
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(title)
        plt.savefig(os.path.join(self.output_path, save_title) + '.png')
        
    def plot_hdbscan_clusters(self, clusterer, X, labels, title, save_title):
        unique_labels = set(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        # HDBSCAN does not have core_sample_indices_, so we use membership probabilities
        membership_strengths = clusterer.probabilities_
        core_samples_mask = membership_strengths > 0.5  # Adjust threshold if needed
        
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                continue
            
            class_member_mask = labels == k
            
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )
            
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.savefig(os.path.join(self.output_path, save_title) + '.png')