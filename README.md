# PMKS-Med
Code for the paper "PMKS-Med: Patient-specific Medication Knowledge Subgraph Enhanced Medication Recommendation".

The code is being updated gradually.

## Data Sets
### Data Request
According to the data-sharing policies of the PhysioNet Clinical Databases, access to the datasets is restricted and requires prior approval. Researchers interested in using the data must submit a formal request through the official PhysioNet platform.

For detailed information and application instructions, please refer to the following pages:

MIMIC-III: https://physionet.org/content/mimiciii/1.4/

MIMIC-IV: https://physionet.org/content/mimiciv/2.2/

The application process is straightforward and designed to ensure responsible data use in accordance with ethical and privacy standards.

### Data Processing
Please process the data according to the data preprocessing pipeline outlined in DEPOT:

🔗 https://github.com/xmed-lab/DrugRec

Additionally, extract the required knowledge graph elements from the UMLS database following the processing workflow provided by UMLSParser:

🔗 https://github.com/DATEXIS/UMLSParser

## RUN
Next, we will take mimic-iii as an example to introduce how to run the code.

Run the following commands in sequence.

### LLM-Guided Rule Mining and PMKS Construction
```
cd 1_ruleMining
python preprocess.py
python path_sampler.py --dataset mimic-iii --max_path_len 3 --anchor 100 --cores 6
python rule_generator.py --dataset mimic-iii --model_name Qwen3-8B -f 50 -l 10
python clean_rule.py --dataset mimic-iii -p Qwen3-8B --model none
python rank_rule.py --dataset mimic-iii -p clean_rules/mimic-iii/Qwen3-8B/none
python gen_kg.py
```

### Base Medication Prediction with EHRs and PMKS
```
cd ../2_deepModelDrugRec
python main.py
```

### PMKS-Guided Probability Refinement with LLMs
```
cd ../3_probabilityCalibration
python sample-triples.py
python main.py
python match.py
python eval.py
```

