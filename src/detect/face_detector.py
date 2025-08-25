import os
from typing import List, Tuple

import cv2


def get_haar_cascade() -> cv2.CascadeClassifier:
	# Use OpenCV's bundled Haar cascade
	cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
	if not os.path.exists(cascade_path):
		raise FileNotFoundError('OpenCV haarcascade not found')
	return cv2.CascadeClassifier(cascade_path)


def detect_faces_bgr(image_bgr, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	cascade = get_haar_cascade()
	faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
	return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def crop_face_bgr(image_bgr, box: Tuple[int, int, int, int], margin: float = 0.2):
	h, w = image_bgr.shape[:2]
	x, y, bw, bh = box
	mx = int(bw * margin)
	my = int(bh * margin)
	x0 = max(0, x - mx)
	y0 = max(0, y - my)
	x1 = min(w, x + bw + mx)
	y1 = min(h, y + bh + my)
	return image_bgr[y0:y1, x0:x1]

