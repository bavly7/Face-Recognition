import os
import argparse
from typing import List, Tuple

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

from src.detect.face_detector import detect_faces_bgr, crop_face_bgr
from src.models.embedding_net import build_model


def preprocess_pil(image: Image.Image, image_size: int = 160):
	transform = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	])
	return transform(image)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--images_root', type=str, required=True)
	parser.add_argument('--checkpoint', type=str, required=True)
	parser.add_argument('--out_path', type=str, required=True)
	parser.add_argument('--embedding_dim', type=int, default=128)
	parser.add_argument('--image_size', type=int, default=160)
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = build_model(embedding_dim=args.embedding_dim).to(device)
	ckpt = torch.load(args.checkpoint, map_location=device)
	model.load_state_dict(ckpt['model'])
	model.eval()

	file_paths: List[str] = []
	embeddings: List[np.ndarray] = []
	face_boxes: List[Tuple[int, int, int, int]] = []

	for root, _, files in os.walk(args.images_root):
		for fname in files:
			if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
				continue
			fpath = os.path.join(root, fname)
			bgr = cv2.imread(fpath)
			if bgr is None:
				continue
			faces = detect_faces_bgr(bgr)
			for box in faces:
				crop = crop_face_bgr(bgr, box)
				if crop.size == 0:
					continue
				pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
				tensor = preprocess_pil(pil, image_size=args.image_size).unsqueeze(0).to(device)
				with torch.no_grad():
					emb = model(tensor).cpu().numpy()[0]
				embeddings.append(emb.astype(np.float32))
				file_paths.append(fpath)
				face_boxes.append(box)

	if len(embeddings) == 0:
		print('No faces found.')
		return

	E = np.stack(embeddings, axis=0)
	paths = np.array(file_paths)
	boxes = np.array(face_boxes)
	os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
	np.savez_compressed(args.out_path, embeddings=E, paths=paths, boxes=boxes)
	print(f'Saved index to {args.out_path} with {len(E)} faces')


if __name__ == '__main__':
	main()

