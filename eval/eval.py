import json
import sklearn.metrics
from collections import Counter
import numpy as np
from alignscore import AlignScore
from summac.model_summac import SummaCConv
from rouge_score import rouge_scorer
from bert_score import score
import evaluate

CATEGORIES = ["EXPERIENCE", "INFORMATION", "CAUSE", "SUGGESTION", "QUESTION"]

def eval_classes(pred, gold):
    y_true = []
    y_pred = []
    for idx in range(len(gold)):
        gold_sample = gold[idx]
        pred_sample = pred[idx]
        if gold_sample["uri"] != pred_sample["uri"]:
            raise Exception("Incorrect submission data format. Please ensure same order as the provided test data file.")
        pred_list = [0, 0, 0, 0, 0]
        true_list = [0, 0, 0, 0, 0]
        for i, cat in enumerate(CATEGORIES):
            if len(gold_sample['spans'][cat]) > 0:
                true_list[i] = 1
            if len(pred_sample['spans'][cat]) > 0:
                pred_list[i] = 1
        y_true.append(true_list)
        y_pred.append(pred_list)
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='macro', zero_division=0.0)
    weighted_f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0.0)
    return macro_f1, weighted_f1

def calculate_overlap(pred_span, gold_span):
    pred_tokens = Counter(pred_span.split())
    gold_tokens = Counter(gold_span.split())
    overlap = sum(min(pred_tokens[token], gold_tokens[token]) for token in pred_tokens if token in gold_tokens)
    return overlap

def calc_proportional_match(pred_spans, gold_spans):
    sample_overlap, sample_len_pred, sample_len_gold = 0, 0, 0
    for cat in CATEGORIES:
        cat_gold_spans = gold_spans[cat]
        cat_pred_spans = pred_spans[cat]
        for pred_span in cat_pred_spans:
            best_overlap = 0
            for gold_span in cat_gold_spans:
                best_overlap = max(best_overlap, calculate_overlap(pred_span, gold_span))
            sample_overlap += best_overlap
            sample_len_pred += len(pred_span.split())
        sample_len_gold += sum(len(x.split()) for x in cat_gold_spans)
    return sample_overlap, sample_len_pred, sample_len_gold

def eval_proportional_match(pred, gold):
    total_overlap, total_len_pred, total_len_gold = 0, 0, 0
    for idx in range(len(gold)):
        gold_sample = gold[idx]
        pred_sample = pred[idx]
        if gold_sample["uri"] != pred_sample["uri"]:
            raise Exception("Incorrect submission data format. Please ensure same order as the provided test data file.")
        sample_overlap, sample_len_pred, sample_len_gold = calc_proportional_match(pred_sample['spans'], gold_sample['spans'])
        total_overlap += sample_overlap
        total_len_pred += sample_len_pred
        total_len_gold += sample_len_gold
    precision = total_overlap / total_len_pred if total_len_pred > 0 else 0
    recall = total_overlap / total_len_gold if total_len_gold > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def calc_strict_match(pred_spans, gold_spans):
    num_correct, num_total_pred, num_total_gold = 0, 0, 0
    for cat in CATEGORIES:
        cat_gold_spans = gold_spans[cat]
        cat_pred_spans = pred_spans[cat]
        for pred_span in cat_pred_spans:
            if pred_span in cat_gold_spans:
                num_correct += 1
        num_total_gold += len(cat_gold_spans)
        num_total_pred += len(cat_pred_spans)
    return num_correct, num_total_pred, num_total_gold

def eval_strict_matching(pred, gold):
    total_correct, total_pred, total_gold = 0, 0, 0
    for idx in range(len(gold)):
        gold_sample = gold[idx]
        pred_sample = pred[idx]
        if gold_sample["uri"] != pred_sample["uri"]:
            raise Exception("Incorrect submission data format. Please ensure same order as the provided test data file.")
        correct_count, pred_count, gold_count = calc_strict_match(pred_sample['spans'], gold_sample['spans'])
        total_correct += correct_count
        total_pred += pred_count
        total_gold += gold_count
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gold if total_gold > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def calc_alignscore(preds, docs):
    """
    Calculates AlignScore. If `preds` contains empty strings, 
    replaces them with 'None' to enable comparison with `docs`.
    """
    preds = [p if p.strip() else "None" for p in preds]

    alignscorer = AlignScore(model='roberta-base', batch_size=16, device='cuda:0', 
                            ckpt_path='/path/to/ckpt', evaluation_mode='nli_sp') # set device='cpu' if cuda not available

    return np.mean(alignscorer.score(contexts=docs, claims=preds))


def cal_summac(preds, docs):
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")
    return np.mean(model_conv.score(docs, preds)['scores'])

def calc_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)
    scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
    return np.mean([s['rouge1'].fmeasure for s in scores]), \
            np.mean([s['rouge2'].fmeasure for s in scores]), \
            np.mean([s['rougeLsum'].fmeasure for s in scores])

def calc_bertscore(preds, refs):
    P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cpu')
    return np.mean(F1.tolist())

def calc_meteor(preds, refs):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=preds, references=refs)
    return results['meteor']

def calc_bleu(preds, refs):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=preds, references=refs)
    return results['bleu']

def prepare_summary_eval(pred, gold):
    preds = []
    refs = []
    refs_bleu = []
    for idx in range(len(gold)):
        gold_sample = gold[idx]
        pred_sample = pred[idx]
        if gold_sample["uri"] != pred_sample["uri"]:
            raise Exception("Incorrect submission data format. Please ensure same order as the provided test data file.")
        for cat in CATEGORIES:
            if len(gold_sample['summaries'][cat]) > 0:
                refs.append(gold_sample['summaries'][cat])
                refs_bleu.append([gold_sample['summaries'][cat]])
                preds.append(pred_sample['summaries'][cat])
    return preds, refs, refs_bleu

def main():
    with open('/path/to/gold', 'r') as file:
        gold_data = json.load(file)
    with open('/path/to/pred', 'r') as file:
        pred_data = json.load(file)

    score_dict = {}

    # Class-level evaluations
    macro_f1, weighted_f1 = eval_classes(pred_data, gold_data)
    score_dict["CLASSIFICATION_Macro_F1"] = macro_f1
    score_dict["CLASSIFICATION_Weighted_F1"] = weighted_f1

    # Span-level evaluations
    precision, recall, f1_score = eval_strict_matching(pred_data, gold_data)
    score_dict["STRICT_MATCHING_P"] = precision
    score_dict["STRICT_MATCHING_R"] = recall
    score_dict["STRICT_MATCHING_F1"] = f1_score
    precision, recall, f1_score = eval_proportional_match(pred_data, gold_data)
    score_dict["PROPORTIONAL_MATCHING_P"] = precision
    score_dict["PROPORTIONAL_MATCHING_R"] = recall
    score_dict["PROPORTIONAL_MATCHING_F1"] = f1_score

    # Summary-level evaluations
    preds, refs, refs_bleu = prepare_summary_eval(pred_data, gold_data)
    rouge1_score, rouge2_score, rougel_score = calc_rouge(preds, refs)
    score_dict['ROUGE1'] = rouge1_score
    score_dict['ROUGE2'] = rouge2_score
    score_dict['ROUGEL'] = rougel_score
    score_dict['BERTScore'] = calc_bertscore(preds, refs)
    score_dict['METEOR'] = calc_meteor(preds, refs)
    score_dict['BLEU'] = calc_bleu(preds, refs_bleu)
    score_dict['AlignScore'] = calc_alignscore(preds, refs)
    score_dict['SummaC'] = cal_summac(preds, refs)

    # Save the results
    with open('/path/to/save', 'w') as f:
        json.dump(score_dict, f, indent=2)

    print(f"Evaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
