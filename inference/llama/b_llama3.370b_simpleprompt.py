import os
import json
from tqdm import tqdm
from typing import List, Dict
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def prepare_task_b_prompt(question: str, context: str, spans: str, perspective: str) -> str:
    return (
        f"You are an expert medical writer and summarizer. "
        f"You receive a user question and a set of labeled answer spans. "
        f"Your task is to generate a focused summary for the specific perspective. \n\n"
        "Important instructions:\n"
        "Only include relevant information from the labeled spans matching the given perspective.\n"
        "Be concise yet comprehensive in capturing key points about that perspective.\n"
        "Do not include extraneous details or perspectives unrelated to the assigned label.\n"
        "If the question or context is relevant, incorporate it to provide clarity.\n\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Spans for {perspective} perspective:\n{spans}\n\n"
        "Only generate one summary for the given perspective. Do not include any other explanations."
        "Final output format:\n"
        "Summary: <Your perspective-focused summary>\n\n"   
    )

def process_task_b_vllm(task_a_results: List[Dict], model_name: str, output_file: str, max_new_tokens=250, batch_size=8):
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=4,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(
            max_tokens=max_new_tokens, 
            temperature=0.7, 
            top_p=0.9,
            )

    results = []

    for i in tqdm(range(0, len(task_a_results), batch_size), desc="Processing Task B"):
        batch = task_a_results[i:i+batch_size]
        prompts = []
        
        for entry in batch:
            question = entry.get("question", "")
            context = entry.get("context", "")
            spans_data = entry.get("spans", {})

            if isinstance(spans_data, list):
                spans_by_label = {}
                for span_entry in spans_data:
                    label = span_entry.get("label", "UNKNOWN").upper()
                    text = span_entry.get("text", "").strip()
                    spans_by_label.setdefault(label, []).append(text)
            elif isinstance(spans_data, dict):
                spans_by_label = spans_data
            else:
                continue

            for label, spans in spans_by_label.items():
                spans_text = "\n".join(spans)
                prompt = prepare_task_b_prompt(question, context, spans_text, label)
                prompts.append({"custom_id": f"{entry['custom_id']}-{label.lower()}", "prompt": prompt})

        responses = llm.generate([p["prompt"] for p in prompts], sampling_params)
        for j, response in enumerate(responses):
            results.append({
                "custom_id": prompts[j]["custom_id"],
                "response": response.outputs[0].text.strip()
            })

    save_jsonl(results, output_file)
    print(f"Task B results saved to {output_file}")

if __name__ == "__main__":
    task_a_results_file = "/task/a/result" 
    model_name = "meta-llama/Llama-3.3-70B-Instruct"  
    output_file = "/path/to/output" 

    task_a_results = load_jsonl(task_a_results_file)

    process_task_b_vllm(task_a_results, model_name, output_file)
