import torch
import torch.nn as nn


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
	# embeddings: [batch, dim], L2-normalized
	# cosine distance -> convert to Euclidean squared since normalized: d^2 = 2 - 2*cos
	sim = embeddings @ embeddings.t()
	dist_sq = 2 - 2 * sim
	# Ensure numerical stability
	return torch.clamp(dist_sq, min=0.0)


class BatchHardTripletLoss(nn.Module):

	def __init__(self, margin: float = 0.2):
		super().__init__()
		self.margin = margin

	def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		dist_sq = pairwise_distances(embeddings)
		labels = labels.view(-1, 1)
		is_pos = (labels == labels.t()).float()
		is_neg = 1.0 - is_pos

		# mask to exclude same indices
		same_idx = torch.eye(labels.size(0), device=labels.device)
		is_pos = is_pos - same_idx

		# hardest positive: max distance among positives
		hard_pos = (dist_sq * is_pos + (1.0 - is_pos) * (-1e6)).max(dim=1).values
		# hardest negative: min distance among negatives
		hard_neg = (dist_sq * is_neg + (1.0 - is_neg) * (1e6)).min(dim=1).values

		loss = torch.relu(hard_pos - hard_neg + self.margin)
		# Only consider anchors with at least one positive
		valid = (is_pos.sum(dim=1) > 0).float()
		loss = (loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)
		return loss

