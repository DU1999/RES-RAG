import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models--sentence-transformers"
model = SentenceTransformer(model_path, device=device)

def topk_similarity(query: str, passages: list[str], threshold: float = 0.5):
    if not passages:
        return []

    with torch.no_grad():
        q = model.encode([query],
                         convert_to_tensor=True,
                         normalize_embeddings=True)  
        D = model.encode(passages,
                         convert_to_tensor=True,
                         normalize_embeddings=True,
                         batch_size=128)  

        scores = (q @ D.T).squeeze(0)     

        mask = scores >= threshold
        if mask.any():
            valid_scores = scores[mask]
            valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            sorted_vals, sort_idx = torch.sort(valid_scores, descending=True)
            sorted_indices = valid_indices[sort_idx]
            results = [(idx.item(), score.item()) for idx, score in zip(sorted_indices, sorted_vals)]
        else:
            best_idx = torch.argmax(scores)
            best_score = scores[best_idx]
            results = [(best_idx.item(), best_score.item())]

    return results

