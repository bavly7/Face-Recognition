import os
import argparse
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.dataset import scan_folder, FacesFolderDataset, default_transform
from src.samplers.balanced_batch_sampler import BalancedBatchSampler
from src.models.embedding_net import build_model
from src.losses.triplet import BatchHardTripletLoss
from src.losses.contrastive import BatchAllContrastiveLoss


def split_train_val(samples: List[Tuple[str, int]], val_ratio: float = 0.1) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
	# Simple per-class split
	by_class = {}
	for path, label in samples:
		by_class.setdefault(label, []).append(path)
	train, val = [], []
	for label, paths in by_class.items():
		paths = sorted(paths)
		n = len(paths)
		val_n = max(1, int(n * val_ratio)) if n > 1 else 0
		val_paths = paths[:val_n]
		train_paths = paths[val_n:]
		for p in train_paths:
			train.append((p, label))
		for p in val_paths:
			val.append((p, label))
	return train, val


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', type=str, required=True)
	parser.add_argument('--out_dir', type=str, default='runs')
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--classes_per_batch', type=int, default=8)
	parser.add_argument('--samples_per_class', type=int, default=8)
	parser.add_argument('--embedding_dim', type=int, default=128)
	parser.add_argument('--loss', type=str, choices=['triplet', 'contrastive'], default='triplet')
	parser.add_argument('--margin', type=float, default=0.2)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--image_size', type=int, default=160)
	parser.add_argument('--val_ratio', type=float, default=0.1)
	parser.add_argument('--num_workers', type=int, default=4)
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	os.makedirs(args.out_dir, exist_ok=True)

	# Data
	samples, idx_to_label = scan_folder(args.data_root)
	assert len(samples) > 0, 'No images found.'
	train_samples, val_samples = split_train_val(samples, val_ratio=args.val_ratio)

	transform = default_transform(args.image_size)
	train_ds = FacesFolderDataset(args.data_root, train_samples, transform=transform)
	val_ds = FacesFolderDataset(args.data_root, val_samples, transform=transform)

	train_labels = [label for _, label in train_samples]
	classes_per_batch = min(args.classes_per_batch, len(set(train_labels)))
	samples_per_class = max(2, min(args.samples_per_class, max(train_labels) + 2))
	batch_sampler = BalancedBatchSampler(train_labels, classes_per_batch, samples_per_class)

	train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	# Model
	model = build_model(embedding_dim=args.embedding_dim).to(device)
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)

	if args.loss == 'triplet':
		criterion = BatchHardTripletLoss(margin=args.margin)
	else:
		criterion = BatchAllContrastiveLoss(margin=args.margin)

	best_val = float('inf')
	best_path = os.path.join(args.out_dir, 'best_model.pt')

	for epoch in range(1, args.epochs + 1):
		model.train()
		running = 0.0
		pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
		for images, labels, _ in pbar:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			emb = model(images)
			loss = criterion(emb, labels)
			opt.zero_grad(set_to_none=True)
			loss.backward()
			opt.step()
			running += loss.item()
			pbar.set_postfix(loss=f'{loss.item():.4f}')

		# Val
		model.eval()
		val_loss = 0.0
		count = 0
		with torch.no_grad():
			for images, labels, _ in tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [val]'):
				images = images.to(device, non_blocking=True)
				labels = labels.to(device, non_blocking=True)
				emb = model(images)
				loss = criterion(emb, labels)
				val_loss += loss.item()
				count += 1
		val_loss = val_loss / max(1, count)
		print(f'Epoch {epoch}: val_loss={val_loss:.4f}')

		if val_loss < best_val:
			best_val = val_loss
			torch.save({'model': model.state_dict(), 'args': vars(args)}, best_path)
			print(f'Saved best model to {best_path}')

	print('Done.')


if __name__ == '__main__':
	main()

