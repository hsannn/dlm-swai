import json
import os
import argparse
import re
import random
from typing import Dict, List, Any
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict


def load_judge_prompt(prompt_path: str) -> str:
    """Load the judge prompt from file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_text_for_evaluation(experiment: str, prompt: str, generated_text: str, subject: str = None) -> str:
    if experiment == "rewriting":
        return generated_text
    elif experiment == "open_ended":
        pattern = r"Continue writing the article about [^\.]+\.\n\n"
        match = re.search(pattern, prompt)
        
        if not match:
            print(f"Warning: Could not find 'Continue writing the article about ...' pattern in prompt. Using generated_text only.")
            return generated_text
        
        continuation_start = match.end()
        original_snippet = prompt[continuation_start:].strip()
        
        if original_snippet:
            combined_text = original_snippet.rstrip() + " " + generated_text.lstrip()
        else:
            combined_text = generated_text
        
        return combined_text
    
    else:
        print(f"Warning: Unknown experiment type '{experiment}'. Using generated_text only.")
        return generated_text


def detect_dataset_type(judge_prompt: str, data: Dict[str, Any]) -> str:

    prompt_lower = judge_prompt.lower()
    if "politeness" in prompt_lower or "polite" in prompt_lower or "impolite" in prompt_lower:
        return "wikipol"
    if "toxic" in prompt_lower or "toxicity" in prompt_lower:
        return "real_tox"

    results = data.get("results", [])
    if results:
        first_sample = results[0]
        label = first_sample.get("label", "").lower()
        if label in ["impolite", "polite", "neutral"]:
            return "wikipol"
        if label in ["toxic", "non-toxic"]:
            return "real_tox"

    # Default to OSE
    return "ose"


def evaluate_single_text_with_gpt(client: OpenAI, text: str, judge_prompt: str, dataset_type: str = "ose", max_retries: int = 3) -> Dict[str, Any]:
    if dataset_type == "wikipol":
        valid_labels = ["POLITE", "IMPOLITE", "NEUTRAL"]
        label_options = "POLITE, IMPOLITE, or NEUTRAL"
        default_label = "NEUTRAL"
        system_message = "You are an evaluator that classifies written requests by their perceived politeness."
    elif dataset_type == "real_tox":
        valid_labels = ["TOXIC", "NON-TOXIC"]
        label_options = "TOXIC or NON-TOXIC"
        default_label = "NON-TOXIC"
        system_message = "You are an evaluator that classifies text by toxicity level."
    else:  # OSE
        valid_labels = ["ELEMENTARY", "INTERMEDIATE", "ADVANCED"]
        label_options = "ELEMENTARY, INTERMEDIATE, or ADVANCED"
        default_label = "INTERMEDIATE"
        system_message = "You are an evaluator that classifies English texts into OneStopEnglish-style levels."

    full_prompt = (
        f"{judge_prompt}\n\n"
        f"Text to evaluate:\n{text}\n\n"
        f'Output ONLY: ["LABEL", confidence] where LABEL is one of {label_options} '
        f"and confidence is a float 0-1. Example: [\"ELEMENTARY\", 0.9]"
    )

    result_text = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": full_prompt},
                ],
            )

            result_text = response.choices[0].message.content.strip()
            parsed = json.loads(result_text)
            if not (isinstance(parsed, list) and len(parsed) == 2):
                raise ValueError(f"Expected [LABEL, confidence], got: {result_text!r}")
            label, conf = str(parsed[0]).strip().upper(), float(parsed[1])
            if label not in valid_labels:
                raise ValueError(f"Invalid label: {label!r}")
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"Confidence out of range: {conf}")

            return {"label": label, "confidence": conf, "reasons": [], "quotes": []}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: Error (attempt {attempt + 1}/{max_retries}): {e}")
            else:
                print(f"Error: Failed after {max_retries} attempts: {e}")

    return {"label": default_label, "confidence": 0.0, "reasons": ["Unknown error"], "quotes": []}


def evaluate_triplet_with_gpt(client: OpenAI, subject: str, text_a: str, text_b: str, text_c: str, judge_prompt: str, dataset_type: str = "ose", max_retries: int = 3) -> Dict[str, Any]:

    if dataset_type == "wikipol":
        valid_labels = ["POLITE", "IMPOLITE", "NEUTRAL"]
        label_options = "POLITE, IMPOLITE, or NEUTRAL"
        default_label = "NEUTRAL"
        system_message = "You are an evaluator that classifies written requests by their perceived politeness."
        context_description = "three rewritten versions of the same original request"
    else:  # OSE
        valid_labels = ["ELEMENTARY", "INTERMEDIATE", "ADVANCED"]
        label_options = "ELEMENTARY, INTERMEDIATE, or ADVANCED"
        default_label = "INTERMEDIATE"
        system_message = "You are an evaluator that classifies English texts into OneStopEnglish-style levels."
        context_description = f'three texts about the same subject: "{subject}"'

    l0, l1, l2 = label_options.split(", ")
    triplet_prompt = (
        f"{judge_prompt}\n\n"
        f"You are given {context_description}.\n\n"
        f"Classify EACH text into exactly one of: {label_options}. "
        f"All three texts must receive different labels (one {l0}, one {l1}, one {l2}).\n\n"
        f"Text A:\n{text_a}\n\nText B:\n{text_b}\n\nText C:\n{text_c}\n\n"
        f'Output ONLY: {{"a":["LABEL",0.9],"b":["LABEL",0.8],"c":["LABEL",0.95]}}'
    )

    result_text = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": triplet_prompt},
                ],
            )

            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            for key in ("a", "b", "c"):
                if key not in result:
                    raise ValueError(f"Missing key: {key}")
                entry = result[key]
                if not (isinstance(entry, list) and len(entry) == 2):
                    raise ValueError(f"Expected [LABEL, confidence] for '{key}', got: {entry!r}")
                label, conf = str(entry[0]).strip().upper(), float(entry[1])
                if label not in valid_labels:
                    raise ValueError(f"Invalid label for '{key}': {label!r}")
                result[key] = (label, conf)

            labels = {result["a"][0], result["b"][0], result["c"][0]}
            if len(labels) != 3:
                raise ValueError(f"Labels must all differ. Got: {result}")

            def _entry(label, conf):
                return {"label": label, "confidence": conf, "reasons": [], "quotes": []}

            return {
                "text_a": _entry(*result["a"]),
                "text_b": _entry(*result["b"]),
                "text_c": _entry(*result["c"]),
                "overall_confidence": round((result["a"][1] + result["b"][1] + result["c"][1]) / 3, 4),
                "comparison_notes": "",
            }

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: Error (attempt {attempt + 1}/{max_retries}): {e}")
            else:
                print(f"Error: Failed after {max_retries} attempts: {e}")

    default_eval = {"label": default_label, "confidence": 0.0, "reasons": ["Unknown error"], "quotes": []}
    return {
        "text_a": default_eval.copy(),
        "text_b": default_eval.copy(),
        "text_c": default_eval.copy(),
        "overall_confidence": 0.0,
        "comparison_notes": "Evaluation failed",
    }


def evaluate_pair_with_gpt(client: OpenAI, original_text: str, text_a: str, text_b: str, judge_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    system_message = "You are an evaluator that classifies text by toxicity level."
    valid_labels = ["TOXIC", "NON-TOXIC"]

    pair_prompt = (
        f"{judge_prompt}\n\n"
        f"You are given two rewritten versions of the same original text.\n\n"
        f"Original text:\n{original_text}\n\n"
        f"Text A:\n{text_a}\n\nText B:\n{text_b}\n\n"
        f"Classify EACH text as TOXIC or NON-TOXIC. The two texts must receive different labels.\n"
        f'Output ONLY: {{"a":["LABEL",0.9],"b":["LABEL",0.8]}}'
    )

    result_text = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": pair_prompt},
                ],
            )

            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)

            for key in ("a", "b"):
                if key not in result:
                    raise ValueError(f"Missing key: {key}")
                entry = result[key]
                if not (isinstance(entry, list) and len(entry) == 2):
                    raise ValueError(f"Expected [LABEL, confidence] for '{key}', got: {entry!r}")
                label, conf = str(entry[0]).strip().upper(), float(entry[1])
                if label not in valid_labels:
                    raise ValueError(f"Invalid label for '{key}': {label!r}")
                result[key] = (label, conf)

            if result["a"][0] == result["b"][0]:
                raise ValueError(f"Both labels must differ. Got: {result}")

            def _entry(label, conf):
                return {"label": label, "confidence": conf, "reasons": [], "quotes": []}

            return {
                "text_a": _entry(*result["a"]),
                "text_b": _entry(*result["b"]),
                "overall_confidence": round((result["a"][1] + result["b"][1]) / 2, 4),
                "comparison_notes": "",
            }

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: Error (attempt {attempt + 1}/{max_retries}): {e}")
            else:
                print(f"Error: Failed after {max_retries} attempts: {e}")

    default_eval = {"label": "NON-TOXIC", "confidence": 0.0, "reasons": ["Unknown error"], "quotes": []}
    return {
        "text_a": default_eval.copy(),
        "text_b": default_eval.copy(),
        "overall_confidence": 0.0,
        "comparison_notes": "Evaluation failed",
    }


def group_by_subject_and_level(results: List[Dict[str, Any]], dataset_type: str = "ose") -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    
    if dataset_type in ("wikipol", "real_tox"):
        for sample in results:
            original_text = sample.get("original_text", "")
            target_attr = sample.get("target_attribute", "").upper()

            if not target_attr:
                print(f"Warning: Sample missing target_attribute. Skipping. sample_index: {sample.get('sample_index', 'unknown')}")
                continue

            if original_text and target_attr:
                grouped[original_text][target_attr.lower()].append(sample)
    else:
        for sample in results:
            subject = sample.get("subject", "")
            level = sample.get("level", "").lower() if sample.get("level") else ""
            if subject and level:
                grouped[subject][level].append(sample)
    
    return dict(grouped)


def combine_wikipol_files(input_path: str) -> Dict[str, Any]:

    # Check if this looks like a wikipol file
    is_wikipol = "wikipol" in input_path.lower()
    has_suffix = "_I.json" in input_path or "_P.json" in input_path or "_N.json" in input_path
    
    if not is_wikipol or not has_suffix:
        # Not a wikipol file or not in expected format, just load normally
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Try to find the other two files
    base_path = input_path
    for suffix in ["_I.json", "_P.json", "_N.json"]:
        if input_path.endswith(suffix):
            base_path = input_path[:-len(suffix)]
            break
    
    files_to_load = {
        "I": base_path + "_I.json",
        "P": base_path + "_P.json",
        "N": base_path + "_N.json"
    }
    
    all_results = []
    combined_data = None
    loaded_count = 0
    
    for target_attr, file_path in files_to_load.items():
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Will process only available files.")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        
        if combined_data is None:
            combined_data = file_data.copy()
        
        for result in file_data.get("results", []):
            result["target_attribute"] = target_attr
            all_results.append(result)
        
        loaded_count += 1
    
    if combined_data is None or loaded_count == 0:
        # Fallback: just load the original file
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    combined_data["results"] = all_results
    print(f"Combined {loaded_count} wikipol files. Total samples: {len(all_results)}")
    return combined_data


def combine_ose_files(input_path: str) -> Dict[str, Any]:
    """Merge the three per-target OSE files (_E, _I, _A) into one, overriding 'level' from the target suffix.

    Handles both DLM logit-steering filenames (e.g. ose_rewriting_E.json)
    and activation-steering filenames (e.g. ose_rewriting_E_L23P-1.json).
    Falls back to loading the file as-is when the pattern doesn't match.
    """
    match = re.search(r'^(.*?)_(E|I|A)(_L\d+P-?\d+)?\.json$', input_path)
    if not match:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    base_path = match.group(1)
    layer_suffix = match.group(3) or ""

    target_to_level = {"E": "elementary", "I": "intermediate", "A": "advanced"}

    all_results = []
    combined_data = None
    loaded_count = 0

    for target, level in target_to_level.items():
        file_path = f"{base_path}_{target}{layer_suffix}.json"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Will process only available files.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)

        if combined_data is None:
            combined_data = file_data.copy()

        for result in file_data.get("results", []):
            result["level"] = level  # override with steering target
            all_results.append(result)

        loaded_count += 1

    if combined_data is None or loaded_count == 0:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    combined_data["results"] = all_results
    print(f"Combined {loaded_count} OSE files. Total samples: {len(all_results)}")
    return combined_data


def combine_real_tox_files(input_path: str) -> Dict[str, Any]:
    """Merge the two per-target real_tox files (_T, _NT) into one, setting 'target_attribute' per sample."""
    match = re.search(r'^(.*)_(NT|T)(_L\d+P-?\d+)?\.json$', input_path)
    if not match:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    base_path = match.group(1)
    layer_suffix = match.group(3) or ""

    all_results = []
    combined_data = None
    loaded_count = 0

    for target in ["T", "NT"]:
        file_path = f"{base_path}_{target}{layer_suffix}.json"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Will process only available files.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)

        if combined_data is None:
            combined_data = file_data.copy()

        for result in file_data.get("results", []):
            result["target_attribute"] = target
            all_results.append(result)

        loaded_count += 1

    if combined_data is None or loaded_count == 0:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    combined_data["results"] = all_results
    print(f"Combined {loaded_count} real_tox files. Total samples: {len(all_results)}")
    return combined_data


def expand_grouped_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert prepare_eval.py grouped format into a flat results list.

    OSE groups have a "subject" field and use level/subject for grouping.
    WikiPol/real_tox groups use original_text/target_attribute.
    """
    ose_level_map = {"E": "elementary", "I": "intermediate", "A": "advanced"}

    results = []
    for group in data.get("groups", []):
        sample_idx = group["sample_index"]
        original_text = group.get("original_text", "")
        original_class = group.get("original_class", "")
        prompt = group.get("prompt", "")
        subject = group.get("subject", "")  # present only for OSE
        is_ose = bool(subject)

        for key, value in group.items():
            if not key.endswith("_text"):
                continue
            target = key[:-5]  # strip "_text"  →  "E", "I", "NT", etc.

            if is_ose:
                results.append({
                    "sample_index": sample_idx,
                    "subject": subject,
                    "original_text": original_text,
                    "level": ose_level_map.get(target, target.lower()),
                    "prompt": prompt,
                    "generated_text": value,
                })
            else:
                results.append({
                    "sample_index": sample_idx,
                    "original_text": original_text,
                    "label": original_class,
                    "prompt": prompt,
                    "generated_text": value,
                    "target_attribute": target,
                })

    expanded = {k: v for k, v in data.items() if k not in ("groups", "num_groups")}
    expanded["results"] = results
    return expanded


def process_output_file(input_path: str, output_path: str, judge_prompt: str, client: OpenAI, start_subject_idx: int = None, end_subject_idx: int = None, resume: bool = False, debug: bool = False, max_triplets: int = None, use_triplet: bool = None):

    with open(input_path, "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    dataset_type = detect_dataset_type(judge_prompt, temp_data)


    if "groups" in temp_data:
        # Pre-processed grouped format from prepare_eval.py — expand in place
        data = expand_grouped_format(temp_data)
    elif use_triplet and dataset_type == "ose":
        data = combine_ose_files(input_path)
    elif use_triplet and dataset_type == "wikipol":
        data = combine_wikipol_files(input_path)
    elif use_triplet and dataset_type == "real_tox":
        data = combine_real_tox_files(input_path)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    results = data.get("results", [])
    total_samples = len(results)
    
    print(f"Loaded {total_samples} samples from {input_path}")
    
    experiment = data.get("experiment", "rewriting")
    
    print(f"Detected dataset type: {dataset_type}")
    
    if use_triplet:
        print(f"Using triplet evaluation mode")
    else:
        print(f"Using individual evaluation mode")
    
    if not use_triplet:
        total_samples = len(results)
        if start_subject_idx is None:
            start_subject_idx = 0
        if end_subject_idx is None:
            end_subject_idx = total_samples
        
        samples_to_process = results[start_subject_idx:end_subject_idx]
        
        evaluated_indices = set()
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get("results", [])
                    for sample in existing_results:
                        if "judge_evaluation" in sample and sample.get("judge_evaluation"):
                            sample_idx = sample.get("sample_index")
                            if sample_idx is not None:
                                evaluated_indices.add(sample_idx)
                
                print(f"Found {len(evaluated_indices)} already evaluated samples. Resuming...")
            except Exception as e:
                print(f"Warning: Could not load existing output file: {e}")
        
        # Filter out already evaluated samples
        samples_to_process = [s for s in samples_to_process if s.get("sample_index") not in evaluated_indices]
        
        if not samples_to_process:
            print(f"All samples have already been evaluated.")
            return
        
        sampled_indices = None
        sampled_samples_set = None
        if max_triplets is not None and max_triplets < len(samples_to_process):
            random.seed(42)
            samples_to_process = random.sample(samples_to_process, max_triplets)
            sampled_indices = {s.get("sample_index") for s in samples_to_process}
            sampled_samples_set = {(s.get("sample_index"), s.get("target_attribute", "")) for s in samples_to_process}
            print(f"Randomly sampled {max_triplets} samples from {len(results)} total samples")
        
        print(f"Processing {len(samples_to_process)} samples (indices {start_subject_idx} to {end_subject_idx-1})")
        
        sample_key_to_sample = {}
        for s in results:
            sample_idx = s.get("sample_index")
            target_attr = s.get("target_attribute", "")
            if sample_idx is not None:
                key = (sample_idx, target_attr)
                sample_key_to_sample[key] = s
        
        for sample in tqdm(samples_to_process, desc="Evaluating samples"):
            generated_text = sample.get("generated_text", "")
            prompt = sample.get("prompt", "")
            
            if not generated_text:
                print(f"Warning: Sample {sample.get('sample_index', 'unknown')} has no generated_text. Skipping.")
                continue
            
            original_text = sample.get("original_text", "")
            text_to_evaluate = extract_text_for_evaluation(
                experiment=experiment,
                prompt=prompt,
                generated_text=generated_text,
                subject=original_text,
            )
            
            evaluation = evaluate_single_text_with_gpt(
                client=client,
                text=text_to_evaluate,
                judge_prompt=judge_prompt,
                dataset_type=dataset_type
            )
            
            sample["judge_evaluation"] = evaluation.copy()
            sample["judge_evaluation"]["triplet_evaluation"] = False
            
            sample_idx = sample.get("sample_index")
            target_attr = sample.get("target_attribute", "")
            if sample_idx is not None:
                key = (sample_idx, target_attr)
                if key in sample_key_to_sample:
                    original_sample = sample_key_to_sample[key]
                    original_sample["judge_evaluation"] = evaluation.copy()
                    original_sample["judge_evaluation"]["triplet_evaluation"] = False
            
            if dataset_type == "wikipol":
                sample["judge_evaluation"]["target_attribute"] = sample.get("target_attribute", "")
                attr_to_label = {"I": "IMPOLITE", "P": "POLITE", "N": "NEUTRAL"}
                target_attr_upper = sample.get("target_attribute", "").upper()
                sample["judge_evaluation"]["expected_label"] = attr_to_label.get(target_attr_upper, "")

                if sample_idx is not None:
                    key = (sample_idx, target_attr)
                    if key in sample_key_to_sample:
                        original_sample = sample_key_to_sample[key]
                        original_sample["judge_evaluation"]["target_attribute"] = sample.get("target_attribute", "")
                        original_sample["judge_evaluation"]["expected_label"] = attr_to_label.get(target_attr_upper, "")
            elif dataset_type == "real_tox":
                tox_to_label = {"toxic": "TOXIC", "non-toxic": "NON-TOXIC"}
                raw_label = sample.get("label", "").lower()
                sample["judge_evaluation"]["expected_label"] = tox_to_label.get(raw_label, raw_label.upper())

                if sample_idx is not None:
                    key = (sample_idx, target_attr)
                    if key in sample_key_to_sample:
                        original_sample = sample_key_to_sample[key]
                        original_sample["judge_evaluation"]["expected_label"] = sample["judge_evaluation"]["expected_label"]
            else:
                # OSE: set expected_label from the level field
                level = sample.get("level", "")
                level_to_label = {"elementary": "ELEMENTARY", "intermediate": "INTERMEDIATE", "advanced": "ADVANCED"}
                sample["judge_evaluation"]["level"] = level
                sample["judge_evaluation"]["expected_label"] = level_to_label.get(level.lower(), level.upper())

                if sample_idx is not None:
                    key = (sample_idx, target_attr)
                    if key in sample_key_to_sample:
                        original_sample = sample_key_to_sample[key]
                        original_sample["judge_evaluation"]["level"] = level
                        original_sample["judge_evaluation"]["expected_label"] = level_to_label.get(level.lower(), level.upper())

            if sampled_samples_set is not None:
                sampled_results = []
                for s in results:
                    sample_idx = s.get("sample_index")
                    target_attr = s.get("target_attribute", "")
                    key = (sample_idx, target_attr)
                    if key in sampled_samples_set:
                        sampled_results.append(s)
                output_data = {k: v for k, v in data.items() if k != "results"}
                output_data["num_samples"] = len(sampled_results)
                output_data["results"] = [s for s in sampled_results if "judge_evaluation" in s]
            else:
                output_data = {k: v for k, v in data.items() if k != "results"}
                output_data["results"] = [s for s in results if "judge_evaluation" in s]
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation complete. Results saved to {output_path}")
        return
    
    grouped = group_by_subject_and_level(results, dataset_type)
    group_keys = sorted(grouped.keys())
    total_groups = len(group_keys)
    
    print(f"Found {total_groups} unique {'subjects' if dataset_type == 'ose' else 'original texts'}")
    
    if dataset_type == "wikipol":
        required_categories = ["i", "p", "n"]
        group_label = "triplets (with I, P, N)"
    elif dataset_type == "real_tox":
        required_categories = ["t", "nt"]
        group_label = "pairs (with T, NT)"
    else:
        required_categories = None
        group_label = None

    if required_categories is not None:
        complete_groups = [
            key for key in group_keys
            if all(cat in grouped[key] and len(grouped[key][cat]) > 0 for cat in required_categories)
        ]
        print(f"Found {len(complete_groups)} complete {group_label}")

        if max_triplets is not None and max_triplets < len(complete_groups):
            random.seed(42)
            complete_groups = random.sample(complete_groups, max_triplets)
            print(f"Randomly sampled {max_triplets} groups from {len(grouped)} total")

        grouped = {key: grouped[key] for key in complete_groups if key in grouped}
        group_keys = sorted(grouped.keys())
        total_groups = len(group_keys)

    if start_subject_idx is None:
        start_subject_idx = 0
    if end_subject_idx is None:
        end_subject_idx = total_groups

    groups_to_process = group_keys[start_subject_idx:end_subject_idx]

    if dataset_type == "wikipol":
        required_categories = ["i", "p", "n"]
        category_names = {"i": "impolite", "p": "polite", "n": "neutral"}
    elif dataset_type == "real_tox":
        required_categories = ["t", "nt"]
        category_names = {"t": "toxic", "nt": "non-toxic"}
    else:
        required_categories = ["elementary", "intermediate", "advanced"]
        category_names = {cat: cat for cat in required_categories}
    
    evaluated_groups = set()
    previously_judged_results = []
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("results", [])
                previously_judged_results = existing_results  # carry over on save
                existing_dataset_type = detect_dataset_type(judge_prompt, existing_data)
                existing_grouped = group_by_subject_and_level(existing_results, existing_dataset_type)
                if existing_dataset_type == "wikipol":
                    existing_required_cats = ["i", "p", "n"]
                elif existing_dataset_type == "real_tox":
                    existing_required_cats = ["t", "nt"]
                else:
                    existing_required_cats = ["elementary", "intermediate", "advanced"]

                for group_key in existing_grouped:
                    group_samples = existing_grouped[group_key]
                    all_evaluated = all(
                        cat in group_samples and all("judge_evaluation" in s for s in group_samples[cat])
                        for cat in existing_required_cats
                    )
                    if all_evaluated:
                        evaluated_groups.add(group_key)

            print(f"Found {len(evaluated_groups)} already evaluated {'subjects' if dataset_type == 'ose' else 'original texts'}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing output file: {e}")
    
    groups_to_process = [g for g in groups_to_process if g not in evaluated_groups]
    
    if not groups_to_process:
        print(f"All {'subjects' if dataset_type == 'ose' else 'original texts'} have already been evaluated.")
        return
    
    print(f"Processing {len(groups_to_process)} {'subjects' if dataset_type == 'ose' else 'original texts'} (indices {start_subject_idx} to {end_subject_idx-1})")
    
    for group_key in tqdm(groups_to_process, desc="Evaluating groups"):
        group_samples = grouped[group_key]
        
        category_to_samples = {}
        
        for cat in required_categories:
            if cat in group_samples and len(group_samples[cat]) > 0:
                category_to_samples[cat] = group_samples[cat][0]
            else:
                group_name = "Subject" if dataset_type == "ose" else "Original text"
                print(f"Warning: {group_name} '{group_key[:50]}...' missing samples for category '{category_names[cat]}'. Skipping triplet evaluation.")
                category_to_samples = None
                break
        
        if category_to_samples is None:
            continue
        
        texts_to_evaluate = {}
        for cat in required_categories:
            sample = category_to_samples[cat]
            generated_text = sample.get("generated_text", "")
            prompt = sample.get("prompt", "")
            
            if not generated_text:
                group_name = "Subject" if dataset_type == "ose" else "Original text"
                print(f"Warning: {group_name} '{group_key[:50]}...' category '{category_names[cat]}' has no generated_text. Skipping.")
                category_to_samples = None
                break
            
            subject_context = group_key if dataset_type == "ose" else sample.get("original_text", "")
            text_to_evaluate = extract_text_for_evaluation(
                experiment=experiment,
                prompt=prompt,
                generated_text=generated_text,
                subject=subject_context,
            )
            texts_to_evaluate[cat] = text_to_evaluate
        
        if category_to_samples is None:
            continue
        
        text_items = [(cat, texts_to_evaluate[cat], category_to_samples[cat]) for cat in required_categories]
        random.shuffle(text_items)

        shuffled_texts = [item[1] for item in text_items]
        text_keys = ["text_a", "text_b", "text_c"][:len(text_items)]
        category_mapping = {text_keys[idx]: cat for idx, (cat, _, _) in enumerate(text_items)}

        # Debug output
        if debug and group_key == groups_to_process[0]:
            print(f"\n{'='*80}")
            group_name = "Subject" if dataset_type == "ose" else "Original text"
            eval_mode = "Pair" if dataset_type == "real_tox" else "Triplet"
            print(f"DEBUG: {group_name} '{group_key[:50]}...' - {eval_mode} evaluation:")
            print(f"{'='*80}")
            print(f"Experiment: {experiment}, Dataset: {dataset_type}")
            for idx, (cat, text, _) in enumerate(text_items):
                text_label = ["Text A", "Text B", "Text C"][idx]
                print(f"{text_label} (shuffled, original category: {category_names[cat]}) length: {len(text)} chars")
                print(f"  First 200 chars: {text[:200]}...")
            print(f"{'='*80}\n")
        
        original_text_for_eval = category_to_samples[required_categories[0]].get("original_text", "")
        if dataset_type == "real_tox":
            evaluation = evaluate_pair_with_gpt(
                client=client,
                original_text=original_text_for_eval,
                text_a=shuffled_texts[0],
                text_b=shuffled_texts[1],
                judge_prompt=judge_prompt,
            )
        else:
            subject_for_eval = group_key if dataset_type == "ose" else original_text_for_eval
            evaluation = evaluate_triplet_with_gpt(
                client=client,
                subject=subject_for_eval,
                text_a=shuffled_texts[0],
                text_b=shuffled_texts[1],
                text_c=shuffled_texts[2],
                judge_prompt=judge_prompt,
                dataset_type=dataset_type
            )

        for text_key, actual_cat in category_mapping.items():
            if actual_cat in category_to_samples:
                sample = category_to_samples[actual_cat]
                sample["judge_evaluation"] = evaluation[text_key].copy()
                sample["judge_evaluation"]["pair_evaluation" if dataset_type == "real_tox" else "triplet_evaluation"] = True
                sample["judge_evaluation"]["overall_confidence"] = evaluation.get("overall_confidence", 0.0)
                sample["judge_evaluation"]["comparison_notes"] = evaluation.get("comparison_notes", "")

                if dataset_type == "wikipol":
                    sample["judge_evaluation"]["target_attribute"] = sample.get("target_attribute", actual_cat.upper())
                    cat_to_label = {"i": "IMPOLITE", "p": "POLITE", "n": "NEUTRAL"}
                    sample["judge_evaluation"]["expected_label"] = cat_to_label.get(actual_cat, "")
                elif dataset_type == "real_tox":
                    sample["judge_evaluation"]["target_attribute"] = sample.get("target_attribute", actual_cat.upper())
                    cat_to_label = {"t": "TOXIC", "nt": "NON-TOXIC"}
                    sample["judge_evaluation"]["expected_label"] = cat_to_label.get(actual_cat, "")
                else:
                    # OSE: store the level and expected_label
                    level = sample.get("level", actual_cat)
                    sample["judge_evaluation"]["level"] = level
                    level_to_label = {"elementary": "ELEMENTARY", "intermediate": "INTERMEDIATE", "advanced": "ADVANCED"}
                    sample["judge_evaluation"]["expected_label"] = level_to_label.get(level.lower(), level.upper())

        for cat in required_categories:
            if cat in group_samples:
                for sample in group_samples[cat]:
                    if sample != category_to_samples[cat]:
                        if "judge_evaluation" in category_to_samples[cat]:
                            sample["judge_evaluation"] = category_to_samples[cat]["judge_evaluation"].copy()

        newly_judged = [s for s in results if "judge_evaluation" in s]
        output_data = {k: v for k, v in data.items() if k != "results"}
        output_data["results"] = previously_judged_results + newly_judged

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated texts using GPT judge with triplet comparison"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="eval_data/llada8b/eval_ose.json",
        help="Path to input JSON file with generated texts (e.g., eval_data/llada8b/eval_ose.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: auto-generated under gpt_judge_results/ mirroring input path)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="judge_system_prompts/ose_sys_prompt.txt",
        help="Path to judge prompt file (default: sys_prompt.txt)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--start-subject-idx",
        type=int,
        default=None,
        help="Start subject index (inclusive) for processing"
    )
    parser.add_argument(
        "--end-subject-idx",
        type=int,
        default=None,
        help="End subject index (exclusive) for processing"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Disable resume: re-evaluate all subjects from scratch"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information including extracted text for evaluation"
    )
    parser.add_argument(
        "--max-triplets",
        type=int,
        default=500,
        help="Maximum number of triplets to evaluate (for triplet mode) or samples to evaluate (for individual mode). For wikipol individual mode, randomly samples this many samples."
    )
    parser.add_argument(
        "--triplet",
        type=lambda x: x.lower() not in ("false", "0", "no"),
        default=True,
        metavar="True|False",
        help="Use triplet/pair evaluation (default: True)"
    )
    parser.add_argument(
        "--data_idx",
        type=int,
        default=None,
        help="Index of a single result sample to evaluate and print (skips saving to file)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.prompt):
        print(f"Error: Judge prompt file not found: {args.prompt}")
        return

    judge_prompt = load_judge_prompt(args.prompt)
    print(f"Loaded judge prompt from {args.prompt}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        return

    client = OpenAI(api_key=api_key)

    # ---- single-sample debug mode ----
    if args.data_idx is not None:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset_type = detect_dataset_type(judge_prompt, data)
        experiment = data.get("experiment", "rewriting")

        # Support both groups format (eval_data) and flat results format
        if "groups" in data:
            groups = data["groups"]
            if args.data_idx >= len(groups):
                print(f"Error: data_idx {args.data_idx} out of range ({len(groups)} groups)")
                return
            group = groups[args.data_idx]

            print("\n" + "=" * 80)
            print("GROUP INFO:")
            print("=" * 80)
            print(f"  sample_index : {group.get('sample_index', args.data_idx)}")
            print(f"  dataset_type : {dataset_type}")
            if dataset_type == "ose":
                print(f"  subject      : {group.get('subject', '')}")
            print(f"  original_class: {group.get('original_class', '')}")
            print("\n" + "=" * 80)
            print("ORIGINAL TEXT:")
            print("=" * 80)
            print(group.get("original_text", ""))

            # Determine which text keys exist in this group
            text_key_map = {
                "ose":      [("E", "E_text"), ("I", "I_text"), ("A", "A_text")],
                "wikipol":  [("I", "I_text"), ("P", "P_text"), ("N", "N_text")],
                "real_tox": [("T", "T_text"), ("NT", "NT_text")],
            }
            variants = text_key_map.get(dataset_type, [])

            prompt = group.get("prompt", "")
            subject_ctx = group.get("subject", group.get("original_text", ""))

            # Build texts for each variant
            texts_by_target = {}
            for target, text_key in variants:
                generated_text = group.get(text_key, "")
                if not generated_text:
                    print(f"\n[{target}] No generated text found, skipping.")
                    continue
                texts_by_target[target] = extract_text_for_evaluation(
                    experiment=experiment,
                    prompt=prompt,
                    generated_text=generated_text,
                    subject=subject_ctx,
                )

            for target, text in texts_by_target.items():
                print("\n" + "=" * 80)
                print(f"TEXT SENT TO JUDGE  [{target}]:")
                print("=" * 80)
                print(text)

            print("\n" + "=" * 80)
            print("JUDGE RESULT:")
            print("=" * 80)

            if args.triplet:
                # Evaluate all variants together (triplet or pair)
                text_list = list(texts_by_target.items())  # [(target, text), ...]
                random.shuffle(text_list)
                text_keys = ["text_a", "text_b", "text_c"][:len(text_list)]
                category_mapping = {text_keys[i]: tgt for i, (tgt, _) in enumerate(text_list)}
                shuffled_texts = [txt for _, txt in text_list]

                if dataset_type == "real_tox":
                    evaluation = evaluate_pair_with_gpt(
                        client=client,
                        original_text=group.get("original_text", ""),
                        text_a=shuffled_texts[0],
                        text_b=shuffled_texts[1],
                        judge_prompt=judge_prompt,
                    )
                else:
                    evaluation = evaluate_triplet_with_gpt(
                        client=client,
                        subject=subject_ctx,
                        text_a=shuffled_texts[0],
                        text_b=shuffled_texts[1],
                        text_c=shuffled_texts[2],
                        judge_prompt=judge_prompt,
                        dataset_type=dataset_type,
                    )

                for text_key, target in category_mapping.items():
                    print(f"  [{target}] → {json.dumps(evaluation[text_key], ensure_ascii=False)}")
            else:
                # Evaluate each variant individually
                for target, text in texts_by_target.items():
                    evaluation = evaluate_single_text_with_gpt(
                        client=client,
                        text=text,
                        judge_prompt=judge_prompt,
                        dataset_type=dataset_type,
                    )
                    print(f"  [{target}] → {json.dumps(evaluation, ensure_ascii=False)}")

            print("=" * 80)

        else:
            # Flat results format
            results = data.get("results", [])
            if args.data_idx >= len(results):
                print(f"Error: data_idx {args.data_idx} out of range ({len(results)} samples)")
                return

            sample = results[args.data_idx]
            generated_text = sample.get("generated_text", "")
            prompt = sample.get("prompt", "")
            subject = sample.get("subject", sample.get("original_text", ""))

            text_to_evaluate = extract_text_for_evaluation(
                experiment=experiment,
                prompt=prompt,
                generated_text=generated_text,
                subject=subject,
            )

            print("\n" + "=" * 80)
            print("SAMPLE INFO:")
            print("=" * 80)
            print(f"  sample_index : {sample.get('sample_index', args.data_idx)}")
            print(f"  dataset_type : {dataset_type}")
            if dataset_type == "ose":
                print(f"  subject      : {sample.get('subject', '')}")
                print(f"  level        : {sample.get('level', '')}")
            else:
                print(f"  label        : {sample.get('label', '')}")
                print(f"  target_attr  : {sample.get('target_attribute', '')}")

            print("\n" + "=" * 80)
            print("TEXT SENT TO JUDGE:")
            print("=" * 80)
            print(text_to_evaluate)

            print("\n" + "=" * 80)
            print("ORIGINAL TEXT:")
            print("=" * 80)
            print(sample.get("original_text", sample.get("corpus", "")))

            print("\n" + "=" * 80)
            print("RUNNING EVALUATION ...")
            print("=" * 80)

            evaluation = evaluate_single_text_with_gpt(
                client=client,
                text=text_to_evaluate,
                judge_prompt=judge_prompt,
                dataset_type=dataset_type,
            )

            print("\n" + "=" * 80)
            print("JUDGE RESULT:")
            print("=" * 80)
            print(json.dumps(evaluation, ensure_ascii=False, indent=2))
            print("=" * 80)

        return

    # ---- normal batch mode ----
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_filename)

        if 'eval_data' in input_dir:
            output_dir = input_dir.replace('eval_data', 'gpt_judge_results')
        elif 'outputs' in input_dir:
            output_dir = input_dir.replace('outputs', 'gpt_judge_results')
        else:
            output_dir = os.path.join('gpt_judge_results', input_dir) if input_dir else 'gpt_judge_results'

        args.output = os.path.join(output_dir, f"{input_name}_judged_triplet{input_ext}")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    process_output_file(
        input_path=args.input,
        output_path=args.output,
        judge_prompt=judge_prompt,
        client=client,
        start_subject_idx=args.start_subject_idx,
        end_subject_idx=args.end_subject_idx,
        resume=not args.no_resume,
        debug=args.debug,
        max_triplets=args.max_triplets,
        use_triplet=args.triplet,
    )


if __name__ == "__main__":
    main()

