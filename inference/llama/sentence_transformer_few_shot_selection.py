import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Dict, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    processed_data = []
    for entry in data:
        processed_data.append({
            "question": entry.get("question", ""),
            "context": entry.get("context", ""),
            "answers": entry.get("answers", []),
            "labelled_answer_spans": entry.get("labelled_answer_spans", {})  
        })
    
    return processed_data

def combine_text(entry: Dict[str, Any]) -> str:
    question = entry["question"]
    context = entry["context"]
    answers = entry["answers"]
    
    combined = f"Question: {question}\nContext: {context}\nAnswers:\n" + "\n".join(answers)
    return combined

def compute_embeddings(data: List[Dict[str, Any]], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [combine_text(entry) for entry in data]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def select_representative_examples(data: List[Dict[str, Any]], embeddings: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    selected_indices = []
    for cluster in range(k):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) == 0:
            continue

        center = cluster_centers[cluster]
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        selected_indices.append(closest_index)
    
    selected_examples = [data[i] for i in selected_indices]
    return selected_examples

def format_few_shot_examples(examples: List[Dict[str, Any]]) -> str:
    formatted_examples = []
    for ex in examples:
        q = ex["question"].strip()
        c = ex["context"].strip()
        spans = ex["labelled_answer_spans"]

        example_text = f"Question: {q}\nContext: {c}\n"

        for perspective, span_list in spans.items():
            span_texts = [span["txt"] for span in span_list]
            if not span_texts:
                span_texts = ["N/A"]
            joined_spans = ", ".join(span_texts)
            example_text += f"{perspective}:\nSpans: {joined_spans}\n\n"

        formatted_examples.append(example_text)

    return "\n".join(formatted_examples)


if __name__ == "__main__":
    data_file = "/path/to/train/data" 
    data = load_data(data_file)
    

    embeddings = compute_embeddings(data, model_name="all-mpnet-base-v2")
    
    selected_examples = select_representative_examples(data, embeddings, k=3)
    few_shot_str = format_few_shot_examples(selected_examples)
    
    print("Selected Few-shot Examples:\n", few_shot_str)
