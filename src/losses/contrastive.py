import torch
import torch.nn as nn


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
	sim = embeddings @ embeddings.t()
	dist_sq = 2 - 2 * sim
	return torch.clamp(dist_sq, min=0.0)


class BatchAllContrastiveLoss(nn.Module):

	def __init__(self, margin: float = 1.0):
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_sq = pairwise_distances(embeddings)
		labels = labels.view(-1, 1)
		is_same = (labels == labels.t()).float()
		# Exclude diagonal pairs
		diag = torch.eye(labels.size(0), device=labels.device)
		mask = 1.0 - diag
		is_same = is_same * mask

		pos_loss = dist_sq * is_same
		neg_loss = torch.relu(self.margin - torch.sqrt(dist_sq + 1e-12)) ** 2 * (1.0 - is_same)

		num_pos = torch.clamp(is_same.sum(), min=1.0)
		num_neg = torch.clamp((1.0 - is_same).sum() - labels.size(0), min=1.0)
		loss = pos_loss.sum() / num_pos + neg_loss.sum() / num_neg
		return loss

