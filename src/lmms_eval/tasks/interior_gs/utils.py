import os
import json
import numpy as np
import pandas as pd
from loguru import logger as eval_logger
from PIL import Image
from collections import OrderedDict

MCA_QUESTION_TYPES = [
    "relative_direction",
    "relative_distance",
]
NA_QUESTION_TYPES = [
    "absolute_distance",
    "object_count",
    "object_size",
]
OPEN_ENDED_QUESTION_TYPES = [
    "reasoning",
]

# Hardcoded base path for simplicity, match what was used in aggregation
BASE_DATA_PATH = "/mnt/4TB_HDD/scene_understanding/dataset/Interior_GS"

def interior_gs_doc_to_visual(doc):
    scene_dir = doc["images"]
    frames_dir = os.path.join(BASE_DATA_PATH, scene_dir, "frames")
    
    image_paths = []
    for i in range(16):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
        if os.path.exists(frame_path):
            image_paths.append(frame_path)
    
    if not image_paths:
        eval_logger.warning(f"No frames found for scene {scene_dir} at {frames_dir}")
        return []
        
    return [Image.open(p) for p in image_paths]

def interior_gs_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are 16 frames sampled from a video of an interior scene."
    
    q_type = doc.get("question_type")
    
    if q_type in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single number."
        return f"{pre_prompt}\nQuestion: {question}\n{post_prompt}"
    elif q_type in MCA_QUESTION_TYPES:
        options = ""
        if "options" in doc:
             options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the best option directly."
        return f"{pre_prompt}\nQuestion: {question}\n{options}\n{post_prompt}"
    else:
        return f"{pre_prompt}\nQuestion: {question}"

def fuzzy_matching(pred):
    if not pred:
        return ""
    return pred.split(' ')[0].rstrip('.').strip().lower()

def exact_match(pred, target):
    return 1. if fuzzy_matching(pred) == fuzzy_matching(target) else 0.

def abs_dist_norm(pred, target):
    try:
        p = float(pred)
        t = float(target)
        if t == 0:
            return 0. if p == 0 else 1.
        return abs(p - t) / t
    except:
        return 1.0

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    try:
        norm_err = abs_dist_norm(pred, target)
        num_pts = int((end - start) / interval) + 1
        conf_intervs = np.linspace(start, end, num_pts)
        # Accuracy is 1 if error is within threshold (1-threshold)
        # e.g. if threshold is 0.95, error must be <= 0.05
        accuracy = (norm_err <= (1 - conf_intervs)).astype(float)
        return accuracy.mean()
    except:
        return 0.0

def interior_gs_process_results(doc, results):
    prediction = results[0]
    q_type = doc.get("question_type")
    target = str(doc.get("gt_answer"))
    
    score = 0.0
    if q_type in MCA_QUESTION_TYPES:
        score = exact_match(prediction, target)
    elif q_type in NA_QUESTION_TYPES:
        score = mean_relative_accuracy(prediction, target)
    elif q_type == "reasoning":
        score = 1.0 # Placeholder
    elif q_type == "existence":
        score = 1.0
        
    return {
        "interior_gs_results": {
            "score": score,
            "question_type": q_type,
            "prediction": prediction,
            "ground_truth": target
        }
    }

def calculate_reasoning_metrics(refs, hyps):
    """
    Compute corpus-level scores for reasoning tasks using BLEU, CIDEr, ROUGE, and METEOR.
    """
    try:
        from lmms_eval.text_metrics_utils.capeval.bleu.bleu import Bleu
        from lmms_eval.text_metrics_utils.capeval.cider.cider import Cider
        from lmms_eval.text_metrics_utils.capeval.rouge.rouge import Rouge
        from lmms_eval.text_metrics_utils.capeval.meteor.meteor import Meteor
        
        # Refs and hyps should be {key: [string]}
        bleu = Bleu(4).compute_score(refs, hyps)
        cider = Cider().compute_score(refs, hyps)
        rouge_l = Rouge().compute_score(refs, hyps)
        meteor = Meteor().compute_score(refs, hyps)

        summary = {
            "BLEU1": bleu[0][0],
            "BLEU4": bleu[0][3],
            "CIDEr": cider[0],
            "ROUGE_L": rouge_l[0],
            "METEOR": meteor[0],
        }
        return summary

    except ImportError as e:
        eval_logger.warning(f"Could not import evaluation libraries for reasoning: {e}")
        return {
            "BLEU1": 0.0,
            "BLEU4": 0.0,
            "CIDEr": 0.0,
            "ROUGE_L": 0.0,
            "METEOR": 0.0,
        }

def interior_gs_aggregate_results(results):
    df = pd.DataFrame(results)
    if df.empty:
        return 0.0
        
    output = {}
    
    for q_type, group in df.groupby("question_type"):
        if q_type == "existence":
            # Map "yes"/"no" to 1/0 for binary metrics
            y_true = (group["ground_truth"].str.lower() == "yes").astype(int).values
            # Use fuzzy matching to clean model output before comparison
            y_pred = (group["prediction"].apply(fuzzy_matching) == "yes").astype(int).values
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            output[f"{q_type}_accuracy"] = acc
            output[f"{q_type}_precision"] = prec
            output[f"{q_type}_recall"] = rec
            output[f"{q_type}_f1"] = f1
        elif q_type == "reasoning":
            # For reasoning, we need to gather all refs and hyps for corpus-level metrics
            refs = {str(i): [row["ground_truth"].lower()] for i, (_, row) in enumerate(group.iterrows())}
            hyps = {str(i): [row["prediction"].lower()] for i, (_, row) in enumerate(group.iterrows())}
            
            reasoning_metrics = calculate_reasoning_metrics(refs, hyps)
            for k, v in reasoning_metrics.items():
                output[f"{q_type}_{k}"] = v
        else:
            output[f"{q_type}_score"] = group["score"].mean()
    
    eval_logger.info(f"Interior GS detailed scores:\n{json.dumps(output, indent=2)}")
    
    # Return overall mean of all metrics (converted to percentage)
    # Filter out metrics that shouldn't be averaged into the main score if necessary, 
    # but for now we follow the user's pattern of averaging everything in output.values()
    overall = np.mean(list(output.values()))
    return overall * 100.0
