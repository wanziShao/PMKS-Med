import json
import pickle
from typing import List, Dict, Set, Tuple
from subgraph_extraction import id_to_cui
from utils import load_records, load_entity_mapping
from tqdm import tqdm

def load_triples(triples_path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(triples_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

# 加载药物推荐概率数据
def load_prob_data(prob_json_path: str) -> Dict[int, List[Dict]]:
    with open(prob_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prob_data_map = {patient['patient_id']: patient['visits'] for patient in data}
    return prob_data_map

def process_patient_data():
    triples_path = "../datasets/kg.txt"
    records_path = "../datastes/mimic-iii/test.pkl"
    mapping_path = "../datastes/mimic-iii/cui_to_name_map_2hop.json"
    prob_json_path = "../2_deepModelDrugRec/saved_results/mimic-iii/IntentKG_final/rec-real-prob-all.json"
    output_path = "../datastes/mimic-iii/refined_subgraph.json"

    triples = load_triples(triples_path)
    records = load_records(records_path)
    entity_mapping = load_entity_mapping(mapping_path)
    prob_data_map = load_prob_data(prob_json_path)

    # Step 1: Match to patient visits
    patients_data = []
    for patient_idx, patient in tqdm(enumerate(records), total=len(records), desc="Processing patients"):
        visit_list = []
        for visit_idx, visit in enumerate(patient):
            diag_ids, proc_ids, _ = visit
            diag_cuis = [id_to_cui('diag', d) for d in diag_ids]
            proc_cuis = [id_to_cui('proc', p) for p in proc_ids]

            visit_prob_data = prob_data_map.get(patient_idx, [])
            med_cuis = []
            if visit_idx < len(visit_prob_data):
                predicted = visit_prob_data[visit_idx].get('predicted', [])
                high_prob_med_ids = [med_id for med_id, prob in predicted if 0.4 <= prob <= 0.6]
                med_cuis = [id_to_cui('med', m) for m in high_prob_med_ids]

            relevant_cuis = set(diag_cuis + med_cuis + proc_cuis)

            visit_triples_set = set()
            for h, r, t in triples:
                if h in relevant_cuis or t in relevant_cuis:
                    visit_triples_set.add((h, r, t))

            mapped_triples = [
                (entity_mapping.get(h, h), r, entity_mapping.get(t, t))
                for h, r, t in visit_triples_set
            ]
            filtered_triples = list({(h, r, t) for h, r, t in mapped_triples})

            visit_list.append({
                "visit_index": visit_idx,
                "filtered_triples": filtered_triples
            })

        patients_data.append({
            "patient_index": patient_idx,
            "visits": visit_list
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patients_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Output saved to {output_path}")

if __name__ == "__main__":
    process_patient_data()
