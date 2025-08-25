import os
import io
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import cv2

from src.models.embedding_net import build_model
from src.detect.face_detector import detect_faces_bgr, crop_face_bgr


@st.cache_resource
def load_model(checkpoint_path: str, embedding_dim: int = 128):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = build_model(embedding_dim=embedding_dim).to(device)
	ckpt = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	return model, device


def preprocess_pil(image: Image.Image, image_size: int = 160):
	transform = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	])
	return transform(image)


def compute_embedding(model, device, image: Image.Image, image_size: int = 160):
	tensor = preprocess_pil(image, image_size=image_size).unsqueeze(0).to(device)
	with torch.no_grad():
		emb = model(tensor).cpu().numpy()[0]
	return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
	return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
	st.title('Face Search (from-scratch embeddings)')
	st.write('Train a model, index a photo folder, and query by example image.')

	ckpt_path = st.text_input('Checkpoint path', 'runs/best_model.pt')
	index_path = st.text_input('Index path (.npz)', 'runs/index.npz')
	image_size = st.number_input('Image size', 64, 256, 160, 8)
	embedding_dim = st.number_input('Embedding dim', 16, 512, 128, 16)

	if not os.path.exists(ckpt_path) or not os.path.exists(index_path):
		st.warning('Please provide valid checkpoint and index files.')
		return

	model, device = load_model(ckpt_path, embedding_dim=int(embedding_dim))
	index = np.load(index_path, allow_pickle=True)
	E = index['embeddings']
	paths = index['paths']
	boxes = index['boxes']

	uploaded = st.file_uploader('Upload query image (face will be detected)', type=['jpg', 'jpeg', 'png'])
	if uploaded is not None:
		data = uploaded.read()
		pil = Image.open(io.BytesIO(data)).convert('RGB')
		bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
		faces = detect_faces_bgr(bgr)
		if not faces:
			st.error('No face detected in query.')
			return
		crop = crop_face_bgr(bgr, faces[0])
		crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
		q = compute_embedding(model, device, crop_pil, image_size=int(image_size))

		sims = E @ q / (np.linalg.norm(E, axis=1) * (np.linalg.norm(q) + 1e-12))
		k = st.slider('Top-K', 1, 50, 12)
		idxs = np.argsort(-sims)[:k]

		st.subheader('Results')
		for i in idxs:
			st.write(f'score={sims[i]:.3f} - {paths[i]}')
			img = Image.open(str(paths[i]))
			st.image(img, use_column_width=True)


if __name__ == '__main__':
	main()

