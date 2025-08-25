## Face Search (from-scratch embeddings with Triplet/Contrastive Loss)

This project trains a face-embedding model from scratch (no pretrained backbones) using Triplet Loss and Contrastive Loss, builds an index of embeddings for a photo collection, and provides a Streamlit app to search for all images containing a queried identity.

### Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare dataset with the following structure (per-identity folders):
```
data_root/
  person_a/
    img001.jpg
    img002.jpg
  person_b/
    img003.jpg
```

3. Train (Triplet or Contrastive):
```bash
python -m src.train --data_root /path/to/data_root --loss triplet --epochs 10 --batch_size 64
```

4. Index a folder of images (faces auto-detected via Haar cascades):
```bash
python -m src.index_embeddings --images_root /path/to/images_folder --checkpoint runs/best_model.pt --out_path runs/index.npz
```

5. Launch Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

### Notes
- Face detection uses OpenCV Haar cascades (classical, not deep pretrained). Recognition embeddings are trained from scratch.
- The model is a lightweight CNN suitable for CPU training for small datasets. For larger datasets, consider deeper architectures and longer training.
# Face-Recognition