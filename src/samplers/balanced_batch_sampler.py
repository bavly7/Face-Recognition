import random
from collections import defaultdict
from typing import Iterable, List, Dict, Iterator

import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[List[int]]):
	"""Yields indices to form batches with N classes and M samples per class."""

	def __init__(self, labels: List[int], classes_per_batch: int, samples_per_class: int):
		self.labels = labels
		self.classes_per_batch = classes_per_batch
		self.samples_per_class = samples_per_class

		self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
		for idx, label in enumerate(labels):
			self.class_to_indices[int(label)].append(idx)

		self.classes: List[int] = list(self.class_to_indices.keys())

	def __iter__(self) -> Iterator[List[int]]:
		# Shuffle per epoch
		for cls in self.classes:
			random.shuffle(self.class_to_indices[cls])

		class_ptr = {cls: 0 for cls in self.classes}
		while True:
			random.shuffle(self.classes)
			batch_classes = []
			for cls in self.classes:
				if class_ptr[cls] + self.samples_per_class <= len(self.class_to_indices[cls]):
					batch_classes.append(cls)
					if len(batch_classes) == self.classes_per_batch:
						break
			if len(batch_classes) < self.classes_per_batch:
				return
			batch_indices: List[int] = []
			for cls in batch_classes:
				start = class_ptr[cls]
				end = start + self.samples_per_class
				batch_indices.extend(self.class_to_indices[cls][start:end])
				class_ptr[cls] = end
			yield batch_indices

	def __len__(self) -> int:
		min_batches_per_class = [len(v) // self.samples_per_class for v in self.class_to_indices.values()]
		if not min_batches_per_class:
			return 0
		total_batches = sum(min_batches_per_class) // self.classes_per_batch
		return total_batches

