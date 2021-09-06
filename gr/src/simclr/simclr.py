import torch.nn as nn
from simclr.modules.identity import Identity
import torch


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, input_ids_i, attn_masks_i, token_type_ids_i, input_ids_j, attn_masks_j, token_type_ids_j):
        h_i, h_i_p = self.encoder(input_ids_i, attn_masks_i, token_type_ids_i)
        h_j, h_j_p = self.encoder(input_ids_j, attn_masks_j, token_type_ids_j)

        z_i = self.mean_pooling(h_i_p, attn_masks_i)
        z_j = self.mean_pooling(h_j_p, attn_masks_j)
        return h_i, h_j, z_i, z_j

    def mean_pooling(self, emb, att_mask):
        mask = att_mask.unsqueeze(-1).expand(emb.size()).float()
        mask_embeddings = emb * mask
        summed = torch.sum(mask_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask

        return mean_pooled