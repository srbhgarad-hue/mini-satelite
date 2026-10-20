import clip
import faiss
import torch
import os
import numpy as np

def test_search():
    print("Testing CLIP + FAISS Semantic Search...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    index = faiss.read_index("vector_db/spatial.index")
    meta = np.load("vector_db/metadata.npy", allow_pickle=True)

    query = "urban area"
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)

    dist, idxs = index.search(feat.cpu().numpy().astype("float32"), k=3)
    print(f"Top matches for '{query}':")
    for i, idx in enumerate(idxs[0]):
        item = meta[idx]
        print(f"  {i+1}. {item['filename']} (Score: {dist[0][i]*100:.1f}%)")
    assert len(idxs[0]) == 3, "Failed to retrieve top 3 results"
    print("[PASS] Semantic search verified successfully!")

if __name__ == "__main__":
    test_search()
