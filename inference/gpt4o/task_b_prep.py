import os
import json
import csv
import tiktoken

COST_PER_INPUT_TOKEN = 0.00000125  # $1.25 per 1M tokens
COST_PER_OUTPUT_TOKEN = 0.00000500  # $5.00 per 1M tokens

experiment_name = "[gpt4o]_[zero]_[b]_[valid400]_[bestprompt]_[0208]"

BATCH_METADATA_FILE = f"../submission/data/b/{experiment_name}_metadata.json"
CSV_LOG_FILE = f"../submission/data/b/{experiment_name}_cost_log.csv"

def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def prepare_batch_file_task_b_candidate_b1(task_a_results_file, test_json_file, output_file):
    with open(test_json_file, "r", encoding="utf-8") as ftest:
        test_data = json.load(ftest)

    with open(task_a_results_file, "r", encoding="utf-8") as f:
        task_a_results = [json.loads(line) for line in f]

    batch_file = []
    total_input_tokens = 0
    total_output_tokens = 0  

    def extract_index(custom_id_str):
        parts = custom_id_str.split("-")
        return int(parts[-1])

    perspective_prompts = {
        "INFORMATION": {"anchor": "For information purposes", "tone": "Informative, Educational"},
        "CAUSE": {"anchor": "Some of the causes", "tone": "Explanatory, Causal"},
        "SUGGESTION": {"anchor": "It is suggested", "tone": "Advisory, Recommending"},
        "EXPERIENCE": {"anchor": "In user’s experience", "tone": "Personal, Narrative"},
        "QUESTION": {"anchor": "It is inquired", "tone": "Seeking Understanding"}
    }

    for entry in task_a_results:
        custom_id = entry.get("custom_id", "")
        spans = entry.get("spans", [])

        try:
            test_index = extract_index(custom_id)
        except ValueError:
            print(f"[WARN] Could not parse index from {custom_id}")
            continue

        if not (0 <= test_index < len(test_data)):
            print(f"[WARN] Index {test_index} out of range.")
            continue

        question = test_data[test_index].get("question", "")
        context = test_data[test_index].get("context", "")

        spans_by_label = {}
        for span_entry in spans:
            span_text = span_entry.get("text", "").strip()
            label = span_entry.get("label", "").strip()
            if span_text and label:
                spans_by_label.setdefault(label, []).append(span_text)

        for label, label_spans in spans_by_label.items():
            spans_text = "\n".join(label_spans)
            perspective_info = perspective_prompts.get(label.upper(), {})

            user_content = (
                f"Question: {question}\n"
                f"Context: {context}\n"
                f"Perspective: {label}\n\n"
                f"Extracted Spans:\n{spans_text}\n\n"
                "Please generate a perspective-focused summary based on the extracted spans above. "
                "Ensure that every detail in your summary is directly supported by the provided spans."
            )

            system_prompt = (
                "While writing summaries, carefully understand the extracted spans to capture "
                "every essential ideas and significant medical details from the text concerning the "
                "perspective of the annotated spans.\n\n"
                f"### Perspective: {label}\n"
                f"- **Anchor Text:** {perspective_info.get('anchor', '')}\n"
                f"- **Tone:** {perspective_info.get('tone', '')}\n\n"
                "Frame summaries appropriately using the defined anchor text. Do not add any additional "
                "information beyond what is explicitly provided in the document.\n\n"
                "Ensure factual consistency by strictly adhering to the extracted spans.\n\n"
                "Finally, always format your output as:\n\n"
                "Summary: <text>"
            )

            input_prompt = f"{system_prompt}\n\n{user_content}"
            input_token_count = count_tokens(input_prompt)
            output_token_count = 1024 

            total_input_tokens += input_token_count
            total_output_tokens += output_token_count

            batch_entry = {
                "custom_id": f"{custom_id}-{label.lower()}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "max_tokens": 2048,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                }
            }

            batch_file.append(batch_entry)

    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in batch_file:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_cost = (total_input_tokens * COST_PER_INPUT_TOKEN) + (total_output_tokens * COST_PER_OUTPUT_TOKEN)

    return batch_file, total_input_tokens, total_output_tokens, system_prompt, total_cost

# ========================= #
#        Logging Functions
# ========================= #

def log_experiment_json(metadata):
    with open(BATCH_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Experiment metadata saved to {BATCH_METADATA_FILE}")

def log_experiment_csv(experiment_name, total_input_tokens, total_output_tokens, avg_input_tokens, output_token_estimate, input_total_cost, output_total_cost, total_cost):
    file_exists = os.path.isfile(CSV_LOG_FILE)

    with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "experiment name", "total input tokens", "total output tokens",
                "avg input tokens", "estimated avg output tokens",
                "input total cost", "output total cost", "total estimated cost"
            ])

        writer.writerow([
            experiment_name, total_input_tokens, total_output_tokens,
            avg_input_tokens, output_token_estimate,
            input_total_cost, output_total_cost, total_cost
        ])

    print(f"✅ Experiment logged in CSV: {CSV_LOG_FILE}")

# ========================= #
#        Execution
# ========================= #

if __name__ == "__main__":
    task_a_results_file = "../submission/data/a/[gpt4o]_[zero]_[a]_[valid400]_[bestprompt]_[postprocessed]_[0208].jsonl"
    test_json_file = "../data/valid_last_400.json"
    output_file = f"../submission/data/b/{experiment_name}_[batchprep].jsonl"

    batch_file, total_input_tokens, total_output_tokens, prompt_template, total_cost = prepare_batch_file_task_b_candidate_b1(task_a_results_file, test_json_file, output_file)

    avg_input_tokens = total_input_tokens / len(batch_file)
    output_token_estimate = total_output_tokens / len(batch_file)
    input_total_cost = total_input_tokens * COST_PER_INPUT_TOKEN
    output_total_cost = total_output_tokens * COST_PER_OUTPUT_TOKEN

    log_experiment_json({
        "experiment_name": experiment_name,
        "model": "gpt-4o",
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1000,
        "num_samples": len(batch_file),
        "prompt_template": prompt_template,
        "output_file": output_file,
        "metadata_file": BATCH_METADATA_FILE,
        "csv_log_file": CSV_LOG_FILE
    })

    log_experiment_csv(experiment_name, total_input_tokens, total_output_tokens, avg_input_tokens, output_token_estimate, input_total_cost, output_total_cost, total_cost)
