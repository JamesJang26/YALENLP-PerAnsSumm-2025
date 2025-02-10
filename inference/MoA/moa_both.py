import os
import gc
import torch
import json
import time
import logging
import ray
from typing import List, Dict, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

##############################################################################
# For logging
##############################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

##############################################################################
# Helper Functions: Load/Save Data and Prepare Inputs
##############################################################################
def load_valid_data(file_path: str) -> List[Dict[str, Any]]:
    """Load valid data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Utility to save list of dicts as JSON lines."""
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Utility to load JSON lines into a list of dicts."""
    items = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def prepare_inputs(entry: Dict[str, Any], prompt_template: str) -> str:
    """
    Prepare input prompts for the model for Task A (basic example).
    - Concat question, context, answers and merge with prompt_template
    """
    question = entry.get("question", "")
    context = entry.get("context", "")
    answers = entry.get("answers", [])
    combined_text = (
        f"Question: {question}\nContext: {context}\nAnswers:\n" + "\n".join(answers)
    )
    return prompt_template.format(combined_text=combined_text)

##############################################################################
# vLLM Cleanup Function
##############################################################################
def cleanup_vllm(llm):
    """
    GPU mem delete after using vLLM
    Github Issue #1908
    """
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        destroy_distributed_environment
    )

    destroy_model_parallel()
    destroy_distributed_environment()

    if hasattr(llm.llm_engine, "model_executor"):
        del llm.llm_engine.model_executor

    del llm

    gc.collect()
    torch.cuda.empty_cache()

    ray.shutdown()

##############################################################################
# vLLM Inference Function
##############################################################################
def run_model_with_vllm(model_path: str, prompts: List[str], sampling_params):
    """
    Load new model every run and cleanup after finishing batch inference
    """
    from vllm import LLM

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.88,
        tensor_parallel_size=4,
        max_model_len=4096,
    )

    outputs = llm.generate(prompts, sampling_params)
    results = [o.outputs[0].text for o in outputs]

    cleanup_vllm(llm)
    return results

##############################################################################
# OpenAI Inference Function
##############################################################################
def run_model_with_openai(model_name: str, prompts: List[str], sampling_params) -> List[str]:
    from openai import OpenAI
    import os

    # Latest OpenAI package usage for chatcompletion
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    temperature = getattr(sampling_params, "temperature", 0.7)
    top_p = getattr(sampling_params, "top_p", 0.9)
    max_tokens = getattr(sampling_params, "max_tokens", None) or getattr(sampling_params, "max_new_tokens", 512)

    results = []
    for prompt in prompts:
        try:
            # client.chat.completions.create(...)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            choice_text = response.choices[0].message.content
            results.append(choice_text)
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            results.append("OpenAI API Error or Timeout.")
    return results



##############################################################################
# Unified Inference Function
##############################################################################
def run_model(model_path: str, prompts: List[str], sampling_params) -> List[str]:
    """
    if: 
      model is openai:, use openai api
    else:
      run with vLLM
    """
    if model_path.startswith("openai:"):
        # e.g. "openai:gpt-3.5-turbo" -> model name: "gpt-3.5-turbo"
        openai_model_name = model_path.split("openai:")[-1].strip()
        return run_model_with_openai(openai_model_name, prompts, sampling_params)
    else:
        # vLLM
        return run_model_with_vllm(model_path, prompts, sampling_params)

##############################################################################
# run_aggregator_with_vllm -> run_aggregator
##############################################################################
def run_aggregator(aggregator_config: Dict[str, Any],
                   final_layer_outputs: List[str],
                   valid_data: List[Dict[str, Any]],
                   prompt_template: str) -> List[str]:
    """
    aggregate after the final layer is done
    """
    from vllm import SamplingParams

    aggregator_prompts = []
    for entry, output in zip(valid_data, final_layer_outputs):
        question = entry.get("question", "")
        context = entry.get("context", "")
        answers = "\n".join(entry.get("answers", []))

        # aggregator prompt
        combined_text = (
            f"Question: {question}\nContext: {context}\nAnswers:\n{answers}\n"
        )
        aggregator_prompt = prompt_template.format(
            combined_text=combined_text,
            previous_output=output
        )
        aggregator_prompts.append(aggregator_prompt)

    sampling_params = SamplingParams(
        temperature=aggregator_config.get("temperature", 0.5),
        top_p=0.9,
        max_tokens=aggregator_config.get("max_new_tokens", 1000),
    )
    model_path = aggregator_config["model_name"]

    all_results = run_model(model_path, aggregator_prompts, sampling_params)
    return all_results

##############################################################################
# Batch Processing for Task A
##############################################################################
def process_task_a(
    valid_data: List[Dict[str, Any]],
    layer_configs: List[List[Dict[str, Any]]],
    aggregator_config: Dict[str, Any],
    output_file: str,
    layer_prompts: List[str],
    aggregator_prompt: str,
):
    """
    여러 레이어를 처리한 뒤, 마지막에 aggregator를 통해 최종 결과를 얻는 함수 (Task A).
    - (예시) Layer 1 & Layer 2, 그리고 Aggregator
    """
    from vllm import SamplingParams

    ########################################################################
    # 1) layer 1
    ########################################################################
    logger.info("===============")
    logger.info("=== Layer 1 ===")
    logger.info("===============")

    # 1.1) prompt for layer 1
    prompts_layer1 = [
        prepare_inputs(entry, layer_prompts[0]) for entry in valid_data
    ]
    previous_outputs = prompts_layer1

    # 1.2) layer 1 model loop
    layer1_results = []
    models_in_layer1 = layer_configs[0]  
    for model_conf in models_in_layer1:
        sampling_params = SamplingParams(
            temperature=model_conf.get("temperature", 0.7),
            top_p=0.9,
            max_tokens=model_conf.get("max_new_tokens", 1000),
        )
        model_path = model_conf["model_name"]
        results = run_model(model_path, previous_outputs, sampling_params)
        layer1_results.append(results)

    # 1.3) concat results from layer 1
    merged_outputs_layer1 = []
    # e.g.: if model_conf is 3 --> (Expert 1, Expert 2, Expert 3)
    # zip(*layer1_results) => tuple (model1_out, model2_out, model3_out)
    for model_outputs_tuple in zip(*layer1_results):
        combined_str_list = []
        for idx, single_model_output in enumerate(model_outputs_tuple, start=1):
            combined_str_list.append(f"Expert {idx}:\n{single_model_output}")
        # concat
        output_text = "\n\n".join(combined_str_list)
        merged_outputs_layer1.append(output_text)

    logger.info("Layer 1 completed.")

    # save intermediate result
    temp_output_file_layer1 = "./temp_layer_1_outputs.jsonl"
    with open(temp_output_file_layer1, "w") as f:
        for output in merged_outputs_layer1:
            f.write(json.dumps({"output": output}) + "\n")
    torch.cuda.empty_cache()
    time.sleep(1)

    # load back
    with open(temp_output_file_layer1, "r") as f:
        layer1_outputs = [json.loads(line)["output"] for line in f]

    ########################################################################
    # 2) layer 2
    ########################################################################
    logger.info("===============")
    logger.info("=== Layer 2 ===")
    logger.info("===============")

    # 2.1) layer 2 prompt
    prompts_layer2 = []
    for entry, out_text in zip(valid_data, layer1_outputs):
        question = entry.get("question", "")
        context = entry.get("context", "")
        answers = "\n".join(entry.get("answers", []))
        original_content = (
            f"Question: {question}\nContext: {context}\nAnswers:\n{answers}\n"
        )
        
        prompt_2 = layer_prompts[1].format(
            combined_text=original_content,
            previous_output=out_text
        )
        prompts_layer2.append(prompt_2)

    previous_outputs = prompts_layer2

   
    layer2_results = []
    models_in_layer2 = layer_configs[1]
    for model_conf in models_in_layer2:
        sampling_params = SamplingParams(
            temperature=model_conf.get("temperature", 0.7),
            top_p=0.9,
            max_tokens=model_conf.get("max_new_tokens", 500),
        )
        model_path = model_conf["model_name"]
        results = run_model(model_path, previous_outputs, sampling_params)
        layer2_results.append(results)


    merged_outputs_layer2 = []
    for model_outputs_tuple in zip(*layer2_results):
        combined_str_list = []
        for idx, single_model_output in enumerate(model_outputs_tuple, start=1):
            combined_str_list.append(f"Expert {idx}:\n{single_model_output}")
        output_text = "\n\n".join(combined_str_list)
        merged_outputs_layer2.append(output_text)

    logger.info("Layer 2 completed.")


    temp_output_file_layer2 = "./temp_layer_2_outputs.jsonl"
    with open(temp_output_file_layer2, "w") as f:
        for output in merged_outputs_layer2:
            f.write(json.dumps({"output": output}) + "\n")
    torch.cuda.empty_cache()
    time.sleep(1)

    with open(temp_output_file_layer2, "r") as f:
        layer2_outputs = [json.loads(line)["output"] for line in f]

    ########################################################################
    # 3) Aggregator (Task A)
    ########################################################################
    logger.info("==================")
    logger.info("=== Aggregator ===")
    logger.info("==================")

    final_responses = run_aggregator(
        aggregator_config,
        layer2_outputs,  
        valid_data,
        layer_prompts[2] if len(layer_prompts) > 2 else aggregator_prompt
    )

    # Save final result
    with open(output_file, "w") as f:
        for idx, response in enumerate(final_responses):
            entry = {"custom_id": f"task-a-{idx}", "response": response}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Task A completed. Results saved to {output_file}.")

##############################################################################
# Parsing Task A output -> Grouping spans by perspective
##############################################################################
def parse_task_a_output(
    task_a_result_file: str,
    valid_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    1) Read the aggregator's final output from Task A.
    2) Parse each line of the form:
         span: <text>, label: <perspective>
       and group them by perspective.

    3) Return a list of dicts. Example structure:
       [
         {
           "uri": <uri_if_any>,
           "spans": {
             "EXPERIENCE": [...],
             "SUGGESTION": [...],
             "INFORMATION": [...],
             "CAUSE": [...],
             "QUESTION": [...]
           }
         },
         ...
       ]
    Note: We assume valid_data[i]["uri"] exists or you can just attach
          the custom_id. Adapt as needed.
    """
    # Load the final aggregator output from Task A
    task_a_results = load_jsonl(task_a_result_file)

    # We assume the order of `task-a-idx` lines corresponds to the order in `valid_data`.
    # If you need a safer approach, you can parse the integer from "custom_id".
    aggregated_spans = []

    for i, item in enumerate(task_a_results):
        # Attempt to find the matching original item
        original = valid_data[i] if i < len(valid_data) else {}
        uri = original.get("uri", f"task-a-{i}")

        response_text = item["response"]
        # Split by lines, parse lines that match: span: <...>, label: <...>
        perspective_buckets = {
            "EXPERIENCE": [],
            "SUGGESTION": [],
            "INFORMATION": [],
            "CAUSE": [],
            "QUESTION": [],
        }

        for line in response_text.splitlines():
            line = line.strip()
            # example: span: the pain is only..., label: EXPERIENCE
            if line.startswith("span: ") and ", label: " in line:
                # parse
                try:
                    after_span_prefix = line[len("span: "):]
                    if ", label: " in after_span_prefix:
                        text_part, label_part = after_span_prefix.split(", label: ", 1)
                        perspective_label = label_part.strip().upper()
                        if perspective_label in perspective_buckets:
                            perspective_buckets[perspective_label].append(text_part.strip())
                        else:
                            # optional: store in "UNKNOWN" or skip
                            pass
                except Exception as e:
                    logger.warning(f"Could not parse line: {line}  Error: {e}")

        aggregated_spans.append({
            "uri": uri,
            "spans": perspective_buckets
        })

    return aggregated_spans

##############################################################################
# Task B: Summaries from spans
##############################################################################
def process_task_b(
    parsed_spans: List[Dict[str, Any]],
    layer_configs: List[List[Dict[str, Any]]],
    aggregator_config: Dict[str, Any],
    output_file: str,
    layer_prompts: List[str],
    aggregator_prompt: str,
):
    """
    Example pipeline for Task B:
    - We'll do multi-layer approach:
      1) layer 1: each perspective → multiple experts → merges to single text
      2) layer 2: optional evaluation or refinement
      3) aggregator: final summary or final perspective-based summary
    """
    from vllm import SamplingParams

    perspective_keys = ["EXPERIENCE", "SUGGESTION", "INFORMATION", "CAUSE", "QUESTION"]

    # 1) Layer 1
    logger.info("===============")
    logger.info("=== Layer 1 (Task B) ===")
    logger.info("===============")

    prompts_layer1 = []
    for item in parsed_spans:
        spans_text = []
        for p in perspective_keys:
            if item["spans"].get(p):
                p_text = "\n".join(item["spans"][p])
                spans_text.append(f"{p} Spans:\n{p_text}")
        all_spans_str = "\n\n".join(spans_text)
        prompt_str = layer_prompts[0].format(all_spans=all_spans_str)
        prompts_layer1.append(prompt_str)

    layer1_results_all_models = []
    models_in_layer1 = layer_configs[0]  # e.g., multiple models
    for model_conf in models_in_layer1:
        sampling_params = SamplingParams(
            temperature=model_conf.get("temperature", 0.7),
            top_p=0.9,
            max_tokens=model_conf.get("max_new_tokens", 500),
        )
        model_path = model_conf["model_name"]
        results = run_model(model_path, prompts_layer1, sampling_params)
        layer1_results_all_models.append(results)

    merged_outputs_layer1 = []
    for model_outputs_tuple in zip(*layer1_results_all_models):
        combined_str_list = []
        for idx, single_model_output in enumerate(model_outputs_tuple, start=1):
            combined_str_list.append(f"Expert {idx}:\n{single_model_output}")
        output_text = "\n\n".join(combined_str_list)
        merged_outputs_layer1.append(output_text)

    logger.info("Layer 1 (Task B) completed.")

    temp_output_file_layer1 = "./temp_task_b_layer1_outputs.jsonl"
    with open(temp_output_file_layer1, "w") as f:
        for output in merged_outputs_layer1:
            f.write(json.dumps({"output": output}) + "\n")
    torch.cuda.empty_cache()
    time.sleep(1)

    with open(temp_output_file_layer1, "r") as f:
        layer1_outputs = [json.loads(line)["output"] for line in f]

    # 2) Layer 2
    logger.info("===============")
    logger.info("=== Layer 2 (Task B) ===")
    logger.info("===============")

    prompts_layer2 = []
    for item, out_text in zip(parsed_spans, layer1_outputs):
        spans_text = []
        for p in perspective_keys:
            if item["spans"].get(p):
                p_text = "\n".join(item["spans"][p])
                spans_text.append(f"{p} Spans:\n{p_text}")
        all_spans_str = "\n\n".join(spans_text)

        prompt_2 = layer_prompts[1].format(
            all_spans=all_spans_str,
            previous_output=out_text
        )
        prompts_layer2.append(prompt_2)

    layer2_results_all_models = []
    models_in_layer2 = layer_configs[1]
    for model_conf in models_in_layer2:
        sampling_params = SamplingParams(
            temperature=model_conf.get("temperature", 0.7),
            top_p=0.9,
            max_tokens=model_conf.get("max_new_tokens", 500),
        )
        model_path = model_conf["model_name"]
        results = run_model(model_path, prompts_layer2, sampling_params)
        layer2_results_all_models.append(results)

    merged_outputs_layer2 = []
    for model_outputs_tuple in zip(*layer2_results_all_models):
        combined_str_list = []
        for idx, single_model_output in enumerate(model_outputs_tuple, start=1):
            combined_str_list.append(f"Expert {idx}:\n{single_model_output}")
        output_text = "\n\n".join(combined_str_list)
        merged_outputs_layer2.append(output_text)

    logger.info("Layer 2 (Task B) completed.")

    temp_output_file_layer2 = "./temp_task_b_layer2_outputs.jsonl"
    with open(temp_output_file_layer2, "w") as f:
        for output in merged_outputs_layer2:
            f.write(json.dumps({"output": output}) + "\n")
    torch.cuda.empty_cache()
    time.sleep(1)

    with open(temp_output_file_layer2, "r") as f:
        layer2_outputs = [json.loads(line)["output"] for line in f]

    # 3) Aggregator (Task B)
    logger.info("==================")
    logger.info("=== Aggregator (Task B) ===")
    logger.info("==================")

    aggregator_prompts = []
    for item, out_text in zip(parsed_spans, layer2_outputs):
        aggregator_prompt_str = aggregator_prompt.format(
            perspective_spans=json.dumps(item["spans"], ensure_ascii=False),
            previous_output=out_text
        )
        aggregator_prompts.append(aggregator_prompt_str)

    sampling_params = SamplingParams(
        temperature=aggregator_config.get("temperature", 0.5),
        top_p=0.9,
        max_tokens=aggregator_config.get("max_new_tokens", 500),
    )
    final_summaries = run_model(aggregator_config["model_name"], aggregator_prompts, sampling_params)

    # Write out final results
    output_data = []
    for i, summary in enumerate(final_summaries):
        output_data.append({
            "custom_id": f"task-b-{i}",
            "uri": parsed_spans[i]["uri"],
            "final_summary": summary
        })
    save_jsonl(output_data, output_file)
    logger.info(f"Task B completed. Final results saved to {output_file}.")

##############################################################################
# Example main execution
##############################################################################
if __name__ == "__main__":

    # model_1 = "meta-llama/Llama-3.3-70B-Instruct"
    model_2 = "openai:gpt-4o-mini"

    model_4 = "openai:gpt-4o"

    valid_file = "../data/test_no_label.json"
    output_file_task_a = "/path/to/intermediate/result/a"
    output_file_task_b = "/path/to/intermediate/result/b"

    # -------------------------------------------------------------------------
    # Task A layer config: 2 layer, 3 models for each layer
    # -------------------------------------------------------------------------
    layer_configs_task_a = [
        [
            {"model_name": model_2, "temperature": 1.2, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.8, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.4, "max_new_tokens": 500},
        ],
        [
            {"model_name": model_2, "temperature": 1.0, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.7, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.4, "max_new_tokens": 500},
        ],
    ]

    # -------------------------------------------------------------------------
    # Task A Aggregator
    # -------------------------------------------------------------------------
    aggregator_config_task_a = {
        "model_name": model_4,  
        "temperature": 0.6,
        "max_new_tokens": 1000,
    }

    # -------------------------------------------------------------------------
    # Task A prompt template
    #  - Layer 1 Prompt
    #  - Layer 2 Prompt
    #  - Aggregator Prompt
    # -------------------------------------------------------------------------
    layer_prompts_task_a = [
        # Layer 1 Prompt
        (
            "Identify the spans in the following text that represent specific perspectives. "
            "For each span, provide the exact text of the span and classify it as one of the "
            "following perspectives: INFORMATION, SUGGESTION, EXPERIENCE, CAUSE, or QUESTION. "
            "Only identify spans from the 'Answers' section. Do not identify or classify any spans "
            "from the 'Question' or 'Context' sections. "
            "{combined_text}\n\n"
            "Structure your response as follows: "
            "'span: <span_text>, label: <label>', "
            "repeated for each identified span."
        ),
        # Layer 2 Prompt
        (
            "Evaluate each span from experts.\n"
            "{combined_text}\n\n"
            "Expert's spans:\n{previous_output}\n\n"
            "Evaluate each spans and label them Good, Poor, or Unclear in the format:\n"
            "expert span: <text>, expert label: <category>, identification eval: <...>, classification eval: <...>\n"
            "No additional explanations and repeated for each evaluated spans."
        ),
        # Aggregator Prompt (Task A)
        (
            "IMPORTANT: Finalize the extracted spans with labels.\n"
            "{combined_text}\n\n"
            "Final from experts:\n{previous_output}\n\n"
            "Output only in format as below without any additional text:\n"
            "span: <text>, label: <category>\n"
        ),
    ]

    # -------------------------------------------------------------------------
    # Task B layer config
    # -------------------------------------------------------------------------
    layer_configs_task_b = [
        [  # layer 1
            {"model_name": model_2, "temperature": 1.2, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.8, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.4, "max_new_tokens": 500},
        ],
        [  # layer 2
            {"model_name": model_2, "temperature": 0.9, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.6, "max_new_tokens": 500},
            {"model_name": model_2, "temperature": 0.3, "max_new_tokens": 500},
        ],
    ]

    # -------------------------------------------------------------------------
    # Task B Aggregator
    # -------------------------------------------------------------------------
    aggregator_config_task_b = {
        "model_name": model_4, 
        "temperature": 0.6,
        "max_new_tokens": 1000,
    }

    # -------------------------------------------------------------------------
    # Task B prompt template
    #  - Layer 1 Prompt
    #  - Layer 2 Prompt
    #  - Aggregator Prompt
    # -------------------------------------------------------------------------
    layer_prompts_task_b = [
        # Layer 1 Prompt
        (
            "Below are categorized spans extracted from user answers.\n"
            "Each category corresponds to a distinct perspective: EXPERIENCE, SUGGESTION, INFORMATION, CAUSE, and QUESTION.\n\n"
            "Perspective Spans:\n{all_spans}\n\n"
            "Your task is to generate a concise summary for each perspective.\n"
            "If a category has no spans, leave its summary blank (e.g., 'EXPERIENCE Summary: ').\n\n"
            "The output MUST follow this exact structure:\n"
            "EXPERIENCE Summary: <summary>\n"
            "SUGGESTION Summary: <summary>\n"
            "INFORMATION Summary: <summary>\n"
            "CAUSE Summary: <summary>\n"
            "QUESTION Summary: <summary>\n\n"
            "Do not include any explanations, extra text, or changes to the output format."
        ),
        # Layer 2 Prompt
        (
            "Here are perspective-based summaries generated by experts.\n\n"
            "Original Perspective Spans:\n{all_spans}\n\n"
            "Expert Summaries:\n{previous_output}\n\n"
            "Your task is to refine or merge the provided summaries into a single coherent set.\n"
            "Ensure that the summaries are clear, concise, and reflect the provided spans accurately.\n"
            "If a category has no relevant spans, leave its summary blank.\n\n"
            "The final output MUST adhere to this structure:\n"
            "EXPERIENCE Summary: <summary>\n"
            "SUGGESTION Summary: <summary>\n"
            "INFORMATION Summary: <summary>\n"
            "CAUSE Summary: <summary>\n"
            "QUESTION Summary: <summary>\n\n"
            "Do not include any explanations, additional comments, or changes to the required format."
        ),
    ]

    aggregator_prompt_task_b = (
        "You are an expert medical writer and summarizer."
        "You receive a user question and a set of labeled answer spans."
        "Your task is to generate a focused summary for the specific perspective."
        "Important instructions:\n"
        "Only include relevant information from the labeled spans matching the given perspective.\n"
        "Be concise yet comprehensive in capturing key points about that perspective.\n"
        "If the question or context is relevant, incorporate it to provide clarity.\n\n"
        "Perspective Spans:\n{perspective_spans}\n\n"
        "Experts Summaries:\n{previous_output}\n\n"
        "The final output MUST strictly follow this structure:\n"
        "EXPERIENCE Summary: <summary>\n"
        "SUGGESTION Summary: <summary>\n"
        "INFORMATION Summary: <summary>\n"
        "CAUSE Summary: <summary>\n"
        "QUESTION Summary: <summary>\n\n"
        "Do not include any extra text, explanations, or deviations from this format."
    )

    ############################################################################
    # 1) Load Data
    ############################################################################
    valid_data = load_valid_data(valid_file)

    ############################################################################
    # 2) Task A: Process Batches (Span Extraction)
    ############################################################################
    process_task_a(
        valid_data=valid_data,
        layer_configs=layer_configs_task_a,
        aggregator_config=aggregator_config_task_a,
        output_file=output_file_task_a,
        layer_prompts=layer_prompts_task_a,
        aggregator_prompt=layer_prompts_task_a[-1],  # or separate aggregator prompt
    )

    ############################################################################
    # 3) Parse Task A results -> group spans by perspective
    ############################################################################
    parsed_spans_data = parse_task_a_output(output_file_task_a, valid_data)
    # Save as JSON for inspection if you like
    with open("./data/test/2layer_3llamaeach_gpt4o_aggregator_from_task_a.json", "w") as f:
        json.dump(parsed_spans_data, f, ensure_ascii=False, indent=2)

    ############################################################################
    # 4) Task B: Summaries from the grouped spans
    ############################################################################
    process_task_b(
        parsed_spans=parsed_spans_data,
        layer_configs=layer_configs_task_b,
        aggregator_config=aggregator_config_task_b,
        output_file=output_file_task_b,
        layer_prompts=layer_prompts_task_b,
        aggregator_prompt=aggregator_prompt_task_b
    )

    logger.info("All done.")
