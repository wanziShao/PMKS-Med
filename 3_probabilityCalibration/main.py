# 只输入0.x~0.y的med及其概率，让LLM修改，其他的药物概率不变
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import pickle
import random
import re
from typing import List, Dict, Any, Set
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 导入工具函数
from utils import load_records, load_entity_mapping, get_med_names_for_indices
from subgraph_extraction import id_to_cui

# Specify local model path and GPU
MODEL_NAME = "/path/to/Qwen3-8B"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ================= 配置区域 =================
# 修改了缓存文件名，防止读取到旧逻辑的数据
CACHE_FILE = "../datastes/mimic-iii/mined_triples_cache_all_meds.json"
# ===========================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# Helper function: Format patient records into readable text
def build_ehr_section(visits_named: List[Dict[str, List[str]]]) -> str:
    lines = []
    for idx, visit in enumerate(visits_named, start=1):
        diag_str = ", ".join(visit["diagnosis"]) if visit["diagnosis"] else "none"
        proc_str = ", ".join(visit["procedure"]) if visit["procedure"] else "none"
        lines.append(f"diagnosis: {diag_str}")
        lines.append(f"procedure: {proc_str}")
    return "\n".join(lines)

# Helper function: Format triples into readable text
def build_kg_section(triples: List[List[str]]) -> str:
    lines = []
    for triple in triples:
        lines.append(json.dumps(triple, ensure_ascii=False))
    return "\n".join(lines)

# List all drug names with indices
def build_drug_list_section(all_medications: List[str]) -> str:
    lines = []
    for idx, name in enumerate(all_medications, start=1):
        lines.append(f"{idx}. \"{name}\"")
    return " ".join(lines)

# Feature extraction function for a specific visit
def extract_features(patient, D, P, M, predict_visit_idx=None):
    diag_counts = np.zeros(D)
    proc_counts = np.zeros(P)
    drug_counts = np.zeros(M)
    if predict_visit_idx is None:
        for visit in patient:
            for d in visit[0]:
                diag_counts[d] += 1
            for p in visit[1]:
                proc_counts[p] += 1
            for m in visit[2]:
                drug_counts[m] += 1
    else:
        for i, visit in enumerate(patient):
            if i <= predict_visit_idx:
                for d in visit[0]:
                    diag_counts[d] += 1
                for p in visit[1]:
                    proc_counts[p] += 1
                if i < predict_visit_idx:
                    for m in visit[2]:
                        drug_counts[m] += 1
    return np.concatenate([diag_counts, proc_counts, drug_counts])

# Modified: Convert visit to a combined set of prefixed diagnosis and procedure codes
def visit_to_feature(visit):
    diag_set = {f'd_{d}' for d in visit[0]}
    proc_set = {f'p_{p}' for p in visit[1]}
    return diag_set.union(proc_set)

# Jaccard similarity calculation
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

# Build patient information function
def build_patient_info(patient, entity_mapping, all_medications, predict_visit_idx=None):
    visits_named = []
    if predict_visit_idx is None:
        visits_to_include = patient
    else:
        visits_to_include = patient[:predict_visit_idx + 1]

    for i, visit in enumerate(visits_to_include):
        if i == len(visits_to_include) - 1:
            diag_ids, proc_ids, med_ids = visit
            diag_names = [f"\"{entity_mapping.get(id_to_cui('diag', d), str(d))}\"" for d in diag_ids]
            proc_names = [f"\"{entity_mapping.get(id_to_cui('proc', p), str(p))}\"" for p in proc_ids]
            visits_named.append({
                "diagnosis": diag_names,
                "procedure": proc_names,
            })
    ehr_text = build_ehr_section(visits_named)
    drug_list_text = build_drug_list_section(all_medications)
    return ehr_text, drug_list_text

# Load visit data from rec-real-30-prob.json
def load_visit_prob_data(json_path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    visit_data_map = {}
    for patient in data:
        patient_id = patient['patient_id']
        visit_data_map[patient_id] = patient['visits']
    return visit_data_map

# Convert medication IDs to names with probabilities
def ids_to_names_with_probs(med_probs: List[List[float]], entity_mapping: Dict[str, str]) -> List[str]:
    return [f"(\"{entity_mapping.get(id_to_cui('med', med_id), str(med_id))}\", \"{prob:.2f}\")" for med_id, prob in med_probs]

# Load visit triples from JSON file
def load_visit_triples(json_path: str) -> Dict[int, Dict[int, List[List[str]]]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    triples_dict = {}
    for patient in data:
        patient_idx = patient['patient_index']
        triples_dict[patient_idx] = {}
        for visit in patient['visits']:
            visit_idx = visit['visit_index']
            triples_dict[patient_idx][visit_idx] = visit['filtered_triples']
    return triples_dict

# Get deduplicated triples for a visit
def get_deduplicated_triples(triples: List[List[str]]) -> List[List[str]]:
    unique_triples = set(tuple(triple) for triple in triples)
    return [list(triple) for triple in unique_triples]

# 通用LLM调用函数
def get_llm_response(prompt: str, temperature: float = 0.3, max_new_tokens: int = 1024) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            min_length=0,
            repetition_penalty=1.0,
        )
    input_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[:, input_len:]
    completion_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return completion_text

# ================= Cache Functions =================
def load_triples_cache(filepath):
    if os.path.exists(filepath):
        print(f"Loading mined triples cache from {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_triples_cache(cache, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Cache saved to {filepath}")
# ===================================================

def main():
    records_path = "../datastes/mimic-iii/records_final.pkl"
    mapping_path = "../datastes/mimic-iii/cui_to_name_map.json"
    test_pkl_path = "../datastes/mimic-iii/test.pkl"
    prob_json_path = "../2_deepModelDrugRec/saved_results/mimic-iii/IntentKG_final/rec-real-prob-all.json"
    triples_json_path = "../datastes/mimic-iii/refined_subgraph.json"
    train_txt_path = "../datastes/mimic-iii/train.pkl"


    # Load data
    records = load_records(records_path)
    train_records = records[:int(len(records)*2/3)]

    with open(test_pkl_path, "rb") as f:
        test_patients = pickle.load(f)
    with open(train_txt_path, "rb") as f:
        train_records_all = pickle.load(f)

    visit_data_map = load_visit_prob_data(prob_json_path)
    entity_mapping = load_entity_mapping(mapping_path)
    all_medications = get_med_names_for_indices(mapping_path, id_to_cui, max_idx=130)

    # Initialize Cache
    triples_cache = load_triples_cache(CACHE_FILE)

    # Vocabulary sizes
    diag_vocab, proc_vocab, drug_vocab = set(), set(), set()
    for p in train_records:
        for v in p:
            diag_vocab.update(v[0])
            proc_vocab.update(v[1])
            drug_vocab.update(v[2])
    D, P, M = max(diag_vocab)+1, max(proc_vocab)+1, max(drug_vocab)+1

    # Precompute combined diagnosis and procedure sets for all visits in train.txt
    train_visit_feats = []
    train_visit_indices = []
    for p_idx, p in enumerate(train_records_all):
        for v_idx, v in enumerate(p):
            feat = visit_to_feature(v)
            train_visit_feats.append(feat)
            train_visit_indices.append((p_idx, v_idx))

    try:
        visit_triples_dict = load_visit_triples(triples_json_path)
    except FileNotFoundError:
        print(f"Error: {triples_json_path} not found.")
        exit(1)

    patients_llm_outputs: List[Dict[str, Any]] = []

    # Main Loop
    try:
        for test_idx, patient in tqdm(list(enumerate(test_patients)), total=len(test_patients)):
            num_visits = len(patient)

            # Ensure cache structure exists
            if str(test_idx) not in triples_cache:
                triples_cache[str(test_idx)] = {}

            for k in range(num_visits):
                # 1. Basic Patient Info
                test_ehr_text, _ = build_patient_info(
                    patient, entity_mapping, all_medications, predict_visit_idx=k
                )

                # ==================================================================================
                # PRE-PHASE: Identify Current Patient's Features & Candidate Meds (0.4-0.6)
                # ==================================================================================

                # Get Candidate Medications
                visit_data_list = visit_data_map.get(test_idx, [])
                if k < len(visit_data_list):
                    visit_data = visit_data_list[k]
                    predicted = visit_data.get('predicted', [])
                else:
                    predicted = []

                selected_meds = [med for med in predicted if 0.4 < med[1] < 0.6]
                top_meds_str = ", ".join(ids_to_names_with_probs(selected_meds, entity_mapping)) if selected_meds else "None"

                # Get Current Diag/Proc Names (for filtering)
                current_diag_ids = patient[k][0]
                current_proc_ids = patient[k][1]
                current_diag_names = {f"\"{entity_mapping.get(id_to_cui('diag', d), str(d))}\"" for d in current_diag_ids}
                current_proc_names = {f"\"{entity_mapping.get(id_to_cui('proc', p), str(p))}\"" for p in current_proc_ids}

                # ==================================================================================
                # PHASE 1: Triples Extraction (Check Cache First)
                # ==================================================================================
                mined_triples_clean = ""

                if str(k) in triples_cache[str(test_idx)]:
                    mined_triples_clean = triples_cache[str(test_idx)][str(k)]
                else:
                    # MISS: Run Similarity Search & Filtering
                    test_visit_feat = visit_to_feature(patient[k])
                    sims = [jaccard_similarity(test_visit_feat, train_feat) for train_feat in train_visit_feats]
                    sims = np.array(sims)

                    sorted_indices = np.argsort(sims)[::-1]
                    top3_indices = sorted_indices[:3]
                    top3_sims = sims[top3_indices]

                    similar_visits_full_text = ""
                    for i, (p_idx, v_idx) in enumerate([train_visit_indices[idx] for idx in top3_indices]):
                        visit = train_records_all[p_idx][v_idx]

                        # Get Names
                        diag_names = [f"\"{entity_mapping.get(id_to_cui('diag', d), str(d))}\"" for d in visit[0]]
                        proc_names = [f"\"{entity_mapping.get(id_to_cui('proc', p), str(p))}\"" for p in visit[1]]
                        med_names = [f"\"{entity_mapping.get(id_to_cui('med', m), str(m))}\"" for m in visit[2]]

                        # === KEY MODIFICATION: Overlapping Diag/Proc, BUT ALL MEDS ===
                        overlapping_diags = [d for d in diag_names if d in current_diag_names]
                        overlapping_procs = [p for p in proc_names if p in current_proc_names]

                        # Use ALL medications from similar visit (no overlap filtering)
                        all_visit_meds = med_names

                        diag_str = ", ".join(overlapping_diags) if overlapping_diags else "None"
                        proc_str = ", ".join(overlapping_procs) if overlapping_procs else "None"
                        med_str = ", ".join(all_visit_meds) if all_visit_meds else "None"
                        # =============================================================

                        similar_visits_full_text += f"Similar Case {i+1} (Similarity: {top3_sims[i]*100:.2f}%):\n"
                        similar_visits_full_text += f"Overlapping Diagnosis: {diag_str}\n"
                        similar_visits_full_text += f"Overlapping Procedure: {proc_str}\n"
                        similar_visits_full_text += f"All Medications Used in Similar Case: {med_str}\n\n"

                    mining_prompt = f"""/no_think
Role: You are a clinical data specialist assisting a medication recommendation system.

**Input Data:**
**Past Cases:**
{similar_visits_full_text}

Task:
From the past cases, extract clinically meaningful triples that represent valid treatment or support relationships between medications and diagnoses or procedures.

Constraints:
- Only use entities (medications, diagnoses, procedures) that explicitly appear in the input.
- Do NOT introduce new entities.
- Extract only relationships that are medically reasonable and meaningful for medication recommendation.
- Exclude experimental compounds, excipients, electrolytes, and non-therapeutic substances.
- Ignore incidental or background medications that do not reflect a treatment decision.

Allowed Relation Types:
- treats
- controls
- prevents
- perioperative_support_for

Output Format:
- Output a JSON list of triples, for example:
  [
    ["Medication Name", "treats", "Diagnosis Name"],
    ["Medication Name", "perioperative_support_for", "Procedure Name"]
  ]
- Do not include explanations or any additional text.
"""
                    mined_triples_text = get_llm_response(mining_prompt, temperature=0.1, max_new_tokens=512)

                    try:
                        json_match = re.search(r'\[.*\]', mined_triples_text, re.DOTALL)
                        if json_match:
                            mined_triples_clean = json_match.group(0)
                        else:
                            mined_triples_clean = "[]"
                    except:
                        mined_triples_clean = "[]"

                    triples_cache[str(test_idx)][str(k)] = mined_triples_clean

                    if test_idx % 20 == 0:
                        save_triples_cache(triples_cache, CACHE_FILE)


                # ==================================================================================
                # PHASE 2: Probability Calibration
                # ==================================================================================

                triples = visit_triples_dict.get(test_idx, {}).get(k, [])
                unique_triples = get_deduplicated_triples(triples)
                if len(unique_triples) > 800:
                    unique_triples = unique_triples[:800]
                original_kg_text = build_kg_section(unique_triples)

                combined_kg_text = f"{original_kg_text}\n{mined_triples_clean}"
                calibration_prompt = f"""/no_think
Role: You are a **senior clinical pharmacy expert** with expertise in electronic health record (EHR) analysis, similar patient pathway mining, and knowledge graph reasoning.

---

**1. Core Task**
**Evaluate and calibrate the candidate medication probabilities generated by the recommendation model**
- Only adjust medications with initial probabilities in the **0.4–0.6** range
- A calibrated probability > 0.5 will be considered as "recommended"

---

**2. Input Data**
2.1. **Patient Electronic Health Record (EHR)**
- Contains the primary diagnosis and relevant clinical information from the current visit
- Used to assess drug indications

EHR:
{test_ehr_text}


2.2. **Candidate Medications and Initial Probabilities**
- ≥ 0.6: High confidence, no adjustment needed
- ≤ 0.4: Low confidence, no adjustment needed
- **0.4 < p < 0.6: Requires adjustment**

Candidate medications and probabilities:
{top_meds_str}


2.3. **Structured Knowledge Graph Triples**
Format: ["Head Entity", "Relation", "Tail Entity"]
Application rules:
- If a drug is indicated as suitable for the patient's current symptoms, the probability of it or its similar drugs being recommended is increased to above 0.5.
- If it is indicated as not suitable or suitable for an unrelated symptom, the probability of it or its similar drugs being recommended is decreased to below 0.5.
- If the information is irrelevant, no adjustment is made.

Knowledge Graph:
{combined_kg_text}

---

**3. Calibration Principles**
- **Evidence-driven**: Only adjust when supported by EHR, similar patient, or KG evidence
- **Moderate adjustments**:
    - Positive evidence from knowledge graph ⇒ Increase probability
    - Negative evidence ⇒ Decrease probability
- **Probability bounds**: Final probabilities must remain within **[0, 1]**

---

**4. Output Requirements**
- No explanation or reasoning required
- Output **only medications whose probabilities were adjusted**, along with their new probabilities
- **Output format:** ("Drug Name 1", "Adjusted Probability 1"), ("Drug Name 2", "Adjusted Probability 2"), ...

---

Based on the above template, output a list containing **only drug names and their adjusted probabilities**.
"""

                final_recommendation = get_llm_response(calibration_prompt, temperature=0.3, max_new_tokens=2048)

                real_med_ids = patient[k][2]
                real_med_names = [entity_mapping.get(id_to_cui("med", m_id), str(m_id)) for m_id in real_med_ids]

                patients_llm_outputs.append({
                    "patient_index": test_idx,
                    "visit_index": k,
                    "mined_triples": mined_triples_clean,
                    "prompt_phase_2": calibration_prompt,
                    "recommendation": final_recommendation,
                    "real_med_ids": real_med_ids,
                    "real_med_names": real_med_names,
                })

    except KeyboardInterrupt:
        print("\nProcess interrupted! Saving cache...")
        save_triples_cache(triples_cache, CACHE_FILE)
        raise

    save_triples_cache(triples_cache, CACHE_FILE)

    output_path = "../datastes/mimic-iii/LLM_probability_calibration.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(patients_llm_outputs, fout, ensure_ascii=False, indent=2)
    print(f"Saved patient recommendations to {output_path}")

if __name__ == "__main__":
    main()
