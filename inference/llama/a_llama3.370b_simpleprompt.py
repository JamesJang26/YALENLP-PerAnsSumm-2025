import os
import json
import torch
import logging
from tqdm import tqdm
from typing import List, Dict
from vllm import LLM, SamplingParams

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Helper functions
def load_json(file_path: str) -> List[Dict]:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dicts to JSONL file."""
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def save_metadata(metadata: Dict, file_path: str):
    """Save metadata JSON file."""
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=4)

def prepare_inputs(entry: Dict) -> str:
    """Prepare input prompts for the model."""
    question = entry.get("question", "")
    context = entry.get("context", "")
    answers = entry.get("answers", [])
    combined_text = f"Question: {question}\nContext: {context}\nAnswers:\n" + "\n".join(answers)
    return (
        "Identify spans from the text below that reflect a specific perspective "
        "and classify them into one of the following categories: "
        "INFORMATION, SUGGESTION, EXPERIENCE, CAUSE, QUESTION.\n\n"
        f"{combined_text}\n\n"
        "For each span, provide the output in the following format:\n"
        "span: <text>, label: <category>\n"
    )

# Main process function
def process_task_a_vllm(
    data: List[Dict], 
    model_name: str, 
    output_file: str, 
    max_new_tokens=1024, 
    batch_size=16, 
    temperature=0.2, 
    top_p=0.9
):
    # Initialize model
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=4,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    # **프롬프트 뼈대만 저장**
    prompt_template = (
        "Identify spans from the text below that reflect a specific perspective "
        "and classify them into one of the following categories: "
        "INFORMATION, SUGGESTION, EXPERIENCE, CAUSE, QUESTION.\n\n"
        "For each span, provide the output in the following format:\n"
        "span: <text>, label: <category>\n"
    )

    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i+batch_size]
        prompts = [prepare_inputs(entry) for entry in batch]
        outputs = llm.generate(prompts, sampling_params)

        for j, output in enumerate(outputs):
            results.append({
                "custom_id": f"task-a-{i + j}",
                "response": output.outputs[0].text.strip()
            })

    # Save results
    save_jsonl(results, output_file)
    logger.info(f"Task A results saved to {output_file}")

    # Save metadata with **only the prompt template**
    metadata = {
        "model": model_name,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "num_samples": len(data),
        "prompt_template": prompt_template
    }
    metadata_file = output_file.replace(".jsonl", "_metadata.json")
    save_metadata(metadata, metadata_file)
    logger.info(f"Metadata saved to {metadata_file}")

# Execution
if __name__ == "__main__":
    valid_file = "../../data/valid.json"
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    output_file = "../submission/data/a/[llama70b]_[zero]_[simpleprompt]_[task_a]_[v1]_[0207].jsonl"

    data = load_json(valid_file)
    process_task_a_vllm(data, model_name, output_file)
