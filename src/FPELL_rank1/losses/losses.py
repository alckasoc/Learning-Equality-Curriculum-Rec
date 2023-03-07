import torch
from torch import nn

from .utils import cos_sim

class BCEWithLogitsMNR(torch.nn.Module):
    def init(self):
        super().init()
        self.mnr_loss = MultipleNegativesRankingLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        
    def forward(self, z, y):
        z = z.view(-1)
        bce_loss = self.bce_loss(z, y)
        
        return 

# Basically the same as this: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
class MultipleNegativesRankingLoss(torch.nn.Module):

    def init(self):
        super().init()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings_a, embeddings_b, labels=None):
        """
        Compute similarity between a and b.
        Labels have the index of the row number at each row. 
        This indicates that a_i and b_j have high similarity 
        when i==j and low similarity when i!=j.
        """

        similarity_scores = (
            cos_sim(embeddings_a, embeddings_b) * 20.0
        )  # Not too sure why to scale it by 20: https://github.com/UKPLab/sentence-transformers/blob/b86eec31cf0a102ad786ba1ff31bfeb4998d3ca5/sentence_transformers/losses/MultipleNegativesRankingLoss.py#L57

        labels = torch.tensor(
            range(len(similarity_scores)),
            dtype=torch.long,
            device=similarity_scores.device,
        )  # Example a[i] should match with b[i]

        return self.loss_function(similarity_scores, labels)