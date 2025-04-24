#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reward computation functions for claim verification.
Modified to use evidence IDs directly for IoU calculation.
"""
import re
import string


def normalize_answer(s):
    """Normalize text: lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str):
    """Extract the label and evidence IDs from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if not match:
        return None, []
    content = match.group(1).strip()
    # Expect format: "Label: X;Evidence: [e1, e2,...]"
    label_match = re.search(r'Label:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)', content)
    evid_match = re.search(r'Evidence:\s*\[([^\]]*)\]', content)
    label = label_match.group(1) if label_match else None
    evid_ids = []
    if evid_match:
        ids_str = evid_match.group(1).strip()
        evid_ids = [eid.strip() for eid in ids_str.split(',') if eid.strip()]
    return label, evid_ids


def compute_reward(solution_str, ground_truth, format_score=0.):
    """
    Compute combined reward:
      - label_reward: 1 if predicted label matches ground_truth['label'], else 0
      - evidence_reward: IoU of predicted vs true evidence IDs (0-1), only if label correct
      - total reward = label_reward + evidence_reward
    """
    # Extract predicted label and evidence IDs
    pred_label, pred_evid_ids = extract_solution(solution_str)
    true_label = ground_truth.get('label')
    true_evid_ids = ground_truth.get('evidence', [])

    # Label reward
    label_reward = 1 if pred_label == true_label else 0

    # Evidence reward
    evidence_reward = 0.0
    if label_reward == 1 and true_evid_ids:
        set_pred = set(pred_evid_ids)
        set_true = set(true_evid_ids)
        # Compute IoU
        intersection = set_pred & set_true
        union = set_pred | set_true
        if union:
            evidence_reward = len(intersection) / len(union)

    return label_reward + evidence_reward, label_reward
