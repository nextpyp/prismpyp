# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchvision.models as models

def get_resnet50(pretrained):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, zero_init_residual=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = models.resnet50(zero_init_residual=True)
    # Remove the classification head
    model = nn.Sequential(*list(model.children())[:-1])
    return model
    
def get_resnet18(pretrained):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, zero_init_residual=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = models.resnet18(zero_init_residual=True)
    # Remove the classification head
    model = nn.Sequential(*list(model.children())[:-1])
    return model
    
def get_resnet34(pretrained):
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT, zero_init_residual=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = models.resnet34(zero_init_residual=True)
    # Remove the classification head
    model = nn.Sequential(*list(model.children())[:-1])
    return model
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, use_checkpoint=True, pretrained=True):
        super(SimSiam, self).__init__()
        
        self.use_checkpoint = use_checkpoint

        # Create the encoder without the classification head
        if base_encoder == "resnet50":
            self.backbone = get_resnet50(pretrained)
        elif base_encoder == "resnet18":
            self.backbone = get_resnet18(pretrained)
        elif base_encoder == "resnet34":
            self.backbone = get_resnet34(pretrained)
            
        # Create dimension reduction MLP
        self.reducer = nn.Sequential(
            nn.Linear(2048, 1024), # ResNet output dim is 2048
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
        prev_dim = dim  # This is the input dimension to the projector (should be 2048 for unmodified ResNet-50)
        # Define the projector using prev_dim
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # First layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # Second layer
            nn.Linear(prev_dim, dim, bias=False),  # Output layer
            nn.BatchNorm1d(dim, affine=False)  # Output layer without affine parameters
        )

        # Build a 2-layer predictor (takes the output of the projector)
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # Hidden layer
            nn.Linear(pred_dim, dim)  # Output layer
        )

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
            kwargs: array of features like ice thickness, CTF fit, estimated resolution, etc.
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        
        # Extract features from the first image (before projection)
        if self.use_checkpoint:
            f1 = checkpoint(self.backbone, x1, use_reentrant=False)  # Feature embedding for view 1
            f2 = checkpoint(self.backbone, x2, use_reentrant=False)  # Feature embedding for view 2
        else:
            f1 = self.backbone(x1)
            f2 = self.backbone(x2)
        
        # Reshape from ResNet output shape of (B, 2048, 1, 1) to (B, 2048)
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        
        # Reduce the dimensionality of the features
        f1 = self.reducer(f1)
        f2 = self.reducer(f2)
            
        # Project the feature embeddings (f1 and f2) using the projector
        z1 = self.projector(f1)  # Projected feature of view 1
        z2 = self.projector(f2)  # Projected feature of view 2

        # Predict using the predictor
        p1 = self.predictor(z1)  # Predicted projection of view 1
        p2 = self.predictor(z2)  # Predicted projection of view 2

        return p1, p2, z1.detach(), z2.detach(), f1, f2
    
    def print_shape_hook(self, module, input, output):
        print(f"Layer: {module.__class__.__name__} | Output shape: {output.shape}")
        
    def get_dissimilarity_matrix(self, metric):
        """
        Input:
            metric: a list of metric for each image in the batch
        Output:
            metric: a NxN matrix where N is the number of images in the batch
        """
        similarity = torch.zeros(len(metric), len(metric))
        for i in range(len(metric)):
            for j in range(len(metric)):
                similarity[i, j] = (metric[i] - metric[j]) ** 2
        
        return similarity
    
    def get_similarity_matrix(self, dissimilarity_matrix):
        """
        Converts MSE dissimilarity matrix to a similarity matrix.
        Args:
            similarity_matrix (torch.Tensor): The MSE dissimilarity matrix.
        Returns:
            torch.Tensor: A similarity matrix (higher values = more similar).
        """
        similarity_matrix = torch.exp(-dissimilarity_matrix)  # This emphasizes smaller differences
        return similarity_matrix

# Define a simple 3-layer MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)