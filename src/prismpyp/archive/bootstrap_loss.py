import torch
import torch.nn as nn
import torch.nn.functional as F

class BootstrapLoss(nn.Module):
    def __init__(self, beta=0.8, soft_labels=True):
        """
        Bootstrap Loss for noisy labels.
        
        Args:
        - beta (float): Weighting factor between target labels and model predictions.
        - soft_labels (bool): If True, use soft bootstrap loss; else, use hard bootstrap.
        """
        super(BootstrapLoss, self).__init__()
        self.beta = beta
        self.soft_labels = soft_labels

    def forward(self, logits, targets):
        """
        Compute Bootstrap Loss.
        
        Args:
        - logits: Model predictions (raw scores before softmax) - shape (batch_size, num_classes)
        - targets: Ground truth labels - shape (batch_size,) for hard labels or (batch_size, num_classes) for one-hot
        
        Returns:
        - loss: Computed bootstrap loss
        """
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        num_classes = logits.shape[1]

        if self.soft_labels:
            # Soft bootstrap loss (blend true labels with predictions)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            modified_targets = self.beta * targets_one_hot + (1 - self.beta) * probs
        else:
            # Hard bootstrap loss (replace incorrect labels with model predictions)
            predicted_labels = probs.argmax(dim=1)
            targets_corrected = torch.where(targets == predicted_labels, targets, predicted_labels)
            modified_targets = F.one_hot(targets_corrected, num_classes=num_classes).float()

        loss = -torch.sum(modified_targets * F.log_softmax(logits, dim=1), dim=1)
        return loss.mean()
