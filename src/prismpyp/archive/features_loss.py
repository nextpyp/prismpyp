import torch
import torch.nn as nn

class IntraInterImageLoss(nn.Module):
    def __init__(self, intra_loss=None, lambda_intra=1.0, lambda_inter=1.0):
        super(IntraInterImageLoss, self).__init__()
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.intra_loss = intra_loss if intra_loss is not None else nn.CosineSimilarity(dim=1)
        # self.inter_loss = inter_loss if inter_loss is not None else nn.CosineSimilarity(dim=1)

    def forward(self, z1, z2, p1, p2, pairwise_similarity, **kwargs):
        # Compute intra-image loss (SimSiam's cosine similarity loss)
        # -(simsiam_loss(p1, z2).mean() + simsiam_loss(p2, z1).mean()) * 0.5
        intra_loss = -(self.intra_loss(p1, z2).mean() + self.intra_loss(p2, z1).mean()) * 0.5
        # Compute inter-image loss (MSE)
        inter_losses = {}
        total_inter_loss = 0.0
        for key, value in kwargs.items():
            inter_losses[key] = torch.square(pairwise_similarity - value.to(z1.device)).mean()
            total_inter_loss += inter_losses[key]
            
        total_loss = self.lambda_intra * intra_loss + self.lambda_inter * total_inter_loss
        return total_loss, intra_loss, total_inter_loss, inter_losses
        