import os
import json
import csv
import tiktoken

COST_PER_INPUT_TOKEN = 0.00000125  # $1.25 per 1M tokens
COST_PER_OUTPUT_TOKEN = 0.00000500  # $5.00 per 1M tokens

experiment_name = "task_a_valid_400_bestprompt_250208"

BATCH_METADATA_FILE = f"../submission/data/a/{experiment_name}_metadata.json"
CSV_LOG_FILE = f"../submission/data/a/{experiment_name}_cost_log.csv"

def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def prepare_batch_file_task_a_candidate_a1(data, experiment_name):
    batch_file = []
    total_input_tokens = 0
    total_output_tokens = 0

    for entry_idx, entry in enumerate(data):
        question = entry.get("question", "")
        context = entry.get("context", "")
        answers = entry.get("answers", [])

        combined_text = (
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Answers:\n" + "\n".join(answers)
        )

        system_prompt = (
            "You are an expert annotator specialized in perspective-aware Healthcare Answer Summarization. "
            "First, validate that the document’s content is aligned with the medical domain – ensure that it pertains to "
            "prevention, diagnosis, management, treatment of diseases, understanding of bodily functions, the effects of "
            "medications or medical interventions, or queries regarding wellness practices.\n\n"
            "Next, for each text span in the 'Answers' section, carefully assess and assign the most relevant perspective(s) "
            "from the following definitions:\n\n"
            "1. INFORMATION: Defined as knowledge about diseases, disorders, and health-related facts, providing insights into "
            "symptoms and diagnosis.\n"
            "2. CAUSE: Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease.\n"
            "3. SUGGESTION: Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, "
            "or improving health issues.\n"
            "4. EXPERIENCE: Defined as individual experiences, anecdotes, or firsthand insights related to health, medical treatments, "
            "medication usage, and coping strategies.\n"
            "5. QUESTION: Defined as an inquiry made for deeper understanding.\n\n"
            "Instructions:\n"
            "- Only annotate spans from the 'Answers' section.\n"
            "- Validate that the document is medically relevant.\n"
            "- For each identified text span, assign the perspective(s) that are most applicable. Multi-perspective labeling is allowed.\n"
            "- If a span explicitly mentions the quantity or duration of medicine ingestion, assign the relevant perspective as well.\n"
            "- Avoid personal bias and do not annotate any links or personally identifiable text.\n"
            "- Review your annotations to ensure you have not missed any underlying perspective.\n\n"
            "Return each identified span with the exact text and its label in the following format:\n"
            "  span: <span_text>, label: <label>\n\n"
            "Do not provide extra explanations; strictly follow the output format."
        )

        input_prompt = f"{system_prompt}\n\n{combined_text}"
        input_token_count = count_tokens(input_prompt)
        output_token_count = 4096  

        total_input_tokens += input_token_count
        total_output_tokens += output_token_count

        batch_file.append({
            "custom_id": f"task-a-{entry_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "top_p": 1.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined_text}
                ]
            }
        })

    return batch_file, total_input_tokens, total_output_tokens, system_prompt

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
    experiment_name = "task_a_valid_400_bestprompt_250208"
    input_file = "../data/valid_last_400.json"
    output_file = f"../submission/data/a/{experiment_name}.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        valid_data = json.load(f)

    print(f"🔄 Preparing batch input file for Task A ({experiment_name})...")
    batch_file_task_a, total_input_tokens, total_output_tokens, prompt_template = prepare_batch_file_task_a_candidate_a1(valid_data, experiment_name)

    # estimated cost
    avg_input_tokens = total_input_tokens / len(valid_data)
    output_token_estimate = total_output_tokens / len(valid_data)
    input_total_cost = total_input_tokens * COST_PER_INPUT_TOKEN
    output_total_cost = total_output_tokens * COST_PER_OUTPUT_TOKEN
    total_cost = input_total_cost + output_total_cost

    print(f"📊 Total Input Tokens: {total_input_tokens}")
    print(f"📊 Total Output Tokens: {total_output_tokens}")
    print(f"📊 Avg Input Tokens: {avg_input_tokens:.2f}")
    print(f"📊 Estimated Avg Output Tokens: {output_token_estimate:.2f}")
    print(f"💰 Estimated Cost: ${total_cost:.4f}")

    metadata = {
        "experiment_name": experiment_name,
        "model": "gpt-4o",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 4096,
        "num_samples": len(valid_data),
        "prompt_template": prompt_template,
        "output_file": output_file,
        "metadata_file": BATCH_METADATA_FILE,
        "csv_log_file": CSV_LOG_FILE
    }
    log_experiment_json(metadata)

    log_experiment_csv(
        experiment_name, total_input_tokens, total_output_tokens,
        avg_input_tokens, output_token_estimate,
        input_total_cost, output_total_cost, total_cost
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for line in batch_file_task_a:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"✅ Batch input file saved: {output_file}")
