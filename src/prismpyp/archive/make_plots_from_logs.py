import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import re
import argparse
import numpy as np

def main(args):
    path_to_log = args.log
    output_path = args.output
    
    # Read log file
    with open(path_to_log, 'r') as f:
        lines = f.readlines()
    
    # Sample loss string:
    # Epoch: [10][ 0/37]	Time 18.285 (18.285)	Data 17.070 (17.070)	Total Loss 0.3123 (0.3123)	SimSiam Loss -0.7110 (-0.7110)	Total Feature Loss 1.0232 (1.0232)	Ice Thickness Loss 0.2655 (0.2655)	Estimated Resolution Loss 0.4739 (0.4739)	CTF Fit Loss 0.2838 (0.2838)
    pattern = re.compile(r"Epoch: \[(\d+)]\[\s*\d+/\d+].*?Total Loss ([\d.-]+).*?SimSiam Loss ([\d.-]+).*?Total Feature Loss ([\d.-]+).*?Ice Thickness Loss ([\d.-]+).*?Estimated Resolution Loss ([\d.-]+).*?CTF Fit Loss ([\d.-]+)")

    epoch_losses = {}
    feature_losses = {}
    
    for line in lines:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            total_loss = float(match.group(2))
            simsiam_loss = float(match.group(3))
            total_feature_loss = float(match.group(4))
            ice_thickness_loss = float(match.group(5))
            estimated_resolution_loss = float(match.group(6))
            ctf_fit_loss = float(match.group(7))

            # Store per epoch
            if epoch not in epoch_losses:
                epoch_losses[epoch] = {'total_loss': [], 'simsiam_loss': [], 'total_feature_loss': []}
                feature_losses[epoch] = {'ice_thickness': [], 'estimated_resolution': [], 'ctf_fit': []}
            
            epoch_losses[epoch]['total_loss'].append(total_loss)
            epoch_losses[epoch]['simsiam_loss'].append(simsiam_loss)
            epoch_losses[epoch]['total_feature_loss'].append(total_feature_loss)

            feature_losses[epoch]['ice_thickness'].append(ice_thickness_loss)
            feature_losses[epoch]['estimated_resolution'].append(estimated_resolution_loss)
            feature_losses[epoch]['ctf_fit'].append(ctf_fit_loss)
    
    # Compute epoch averages
    epochs = sorted(epoch_losses.keys())
    avg_total_loss = [np.mean(epoch_losses[epoch]['total_loss']) for epoch in epochs]
    avg_simsiam_loss = [np.mean(epoch_losses[epoch]['simsiam_loss']) for epoch in epochs]
    avg_total_feature_loss = [np.mean(epoch_losses[epoch]['total_feature_loss']) for epoch in epochs]

    avg_ice_thickness_loss = [np.mean(feature_losses[epoch]['ice_thickness']) for epoch in epochs]
    avg_estimated_resolution_loss = [np.mean(feature_losses[epoch]['estimated_resolution']) for epoch in epochs]
    avg_ctf_fit_loss = [np.mean(feature_losses[epoch]['ctf_fit']) for epoch in epochs]
    avg_sum_feature_loss = [i + e + c for i, e, c in zip(avg_ice_thickness_loss, avg_estimated_resolution_loss, avg_ctf_fit_loss)]
    
    # Individual loss plots
    print("Plotting total loss, simsiam loss, and total feature loss")
    plot_loss(epochs, avg_total_loss, "Total Loss", "Loss", os.path.join(output_path, "total_loss.png"))
    plot_loss(epochs, avg_simsiam_loss, "SimSiam Loss", "Loss", os.path.join(output_path, "simsiam_loss.png"))
    plot_loss(epochs, avg_total_feature_loss, "Total Feature Loss", "Loss", os.path.join(output_path, "total_feature_loss.png"))

    print("Plotting individual feature losses")
    # Aggregate Feature Loss Plot
    plt.figure()
    plt.plot(epochs, avg_ice_thickness_loss, marker='o', label='Ice Thickness Loss')
    plt.plot(epochs, avg_estimated_resolution_loss, marker='s', label='Estimated Resolution Loss')
    plt.plot(epochs, avg_ctf_fit_loss, marker='^', label='CTF Fit Loss')
    plt.plot(epochs, avg_sum_feature_loss, marker='x', linestyle='dashed', label='Sum of Feature Losses')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Feature Losses Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'feature_losses.png'))
    plt.close()
    
    print("Plotting total loss components")
    # Aggregate simsiam loss and total feature loss with total loss
    plt.figure()
    plt.plot(epochs, avg_total_loss, marker='o', label='Total Loss')
    plt.plot(epochs, avg_simsiam_loss, marker='s', label='SimSiam Loss')
    plt.plot(epochs, avg_total_feature_loss, marker='^', label='Total Feature Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Components Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, 'total_loss_components.png'))
    plt.close()


# Plot functions
def plot_loss(epochs, values, title, ylabel, filename):
    plt.figure()
    plt.plot(epochs, values, marker='o', label=title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots from log files')
    parser.add_argument('log', type=str, help='Path to log file')
    parser.add_argument('output', type=str, help='Path to output directory')
    args = parser.parse_args()
    main(args)