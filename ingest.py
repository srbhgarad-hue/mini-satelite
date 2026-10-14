import os
import glob
import numpy as np
import torch
from PIL import Image
import faiss
import clip

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def process_and_index_images(image_dir="data/raw/", index_save_path="vector_db/spatial.index"):
    image_paths = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_paths:
        print(f"Error: No images found in '{image_dir}' directory!")
        return

    embeddings = []
    metadata = []

    print(f"Processing {len(image_paths)} images...")

    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_feature = model.encode_image(tensor_img)
                img_feature /= img_feature.norm(dim=-1, keepdim=True)
                
            embeddings.append(img_feature.cpu().numpy()[0])
            metadata.append({"id": idx, "path": path, "filename": os.path.basename(path)})
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    if not embeddings:
        print("No valid embeddings were generated.")
        return

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    
    # FAISS Index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    faiss.write_index(index, index_save_path)
    np.save(os.path.join(os.path.dirname(index_save_path), "metadata.npy"), metadata)
    print(f"Indexed {len(embeddings)} images successfully at '{index_save_path}'!")

if __name__ == "__main__":
    process_and_index_images()