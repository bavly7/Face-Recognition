import os
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(image_size: int = 160):
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	])


class FacesFolderDataset(Dataset):
	"""Folder structure: root/label_name/*.jpg"""

	def __init__(self, root: str, samples: List[Tuple[str, int]], transform=None):
		self.root = root
		self.samples = samples
		self.transform = transform or default_transform()

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		path, label = self.samples[index]
		img = Image.open(path).convert("RGB")
		img = self.transform(img)
		return img, torch.tensor(label, dtype=torch.long), path


def scan_folder(root: str) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
	label_to_idx: Dict[str, int] = {}
	idx_to_label: Dict[int, str] = {}
	samples: List[Tuple[str, int]] = []
	for label_name in sorted(os.listdir(root)):
		label_dir = os.path.join(root, label_name)
		if not os.path.isdir(label_dir):
			continue
		if label_name not in label_to_idx:
			idx = len(label_to_idx)
			label_to_idx[label_name] = idx
			idx_to_label[idx] = label_name
		for fname in os.listdir(label_dir):
			fpath = os.path.join(label_dir, fname)
			if not os.path.isfile(fpath):
				continue
			if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
				samples.append((fpath, label_to_idx[label_name]))
	return samples, idx_to_label

