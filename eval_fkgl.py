"""
Compute Flesch–Kincaid Grade Level (FKGL) for text samples.

Features
- Sentence, word, and syllable counting
- FKGL score computation
- Single text mode
- Batch evaluation from .txt files or CSV/JSONL
- Optional output file with per-sample scores

Examples
1) Single text
   python eval_fkgl.py --text "This is a simple example. It is easy to read."

2) Text file
   python eval_fkgl.py --input data/example.txt

3) Directory of .txt files
   python eval_fkgl.py --input data/texts/

4) CSV with a text column
   python eval_fkgl.py --input samples.csv --text-column text --output scores.csv

5) JSONL with a text field
   python eval_fkgl.py --input samples.jsonl --text-column content --output scores.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data if not already present
for _pkg in ("punkt_tab", "cmudict"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if _pkg.startswith("punkt") else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_CMUDICT: Dict[str, List[List[str]]] = nltk.corpus.cmudict.dict()

WORD_RE = re.compile(r"[A-Za-z]+(?:[‘’-][A-Za-z]+)*")


@dataclass
class FKGLResult:
    id: Optional[str]
    num_sentences: int
    num_words: int
    num_syllables: int
    avg_words_per_sentence: float
    avg_syllables_per_word: float
    fkgl: Optional[float]


def split_sentences(text: str) -> List[str]:
    """
    Sentence splitter using nltk.sent_tokenize.
    Handles abbreviations (Dr., U.S.A., etc.) correctly.
    """
    text = text.strip()
    if not text:
        return []
    return [s for s in sent_tokenize(text) if s.strip()]


def extract_words(text: str) -> List[str]:
    """
    Extract word-like tokens for English FKGL computation.
    """
    return WORD_RE.findall(text)


def _heuristic_syllables(word: str) -> int:
    """
    Fallback heuristic syllable counter for words not in cmudict.
    """
    word = re.sub(r"[^a-z]", "", word.lower().strip())
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    if word.endswith("e"):
        word = word[:-1]

    syllables = len(re.findall(r"[aeiouy]+", word))

    if re.search(r"[^aeiouy]l$", word):
        syllables += 1

    return max(1, syllables)


def count_syllables(word: str) -> int:
    """
    Count syllables using nltk cmudict; falls back to heuristic for unknown words.
    """
    key = word.lower()
    entries = _CMUDICT.get(key)
    if entries:
        # Count phonemes that end with a digit (stressed/unstressed vowel markers)
        return sum(1 for ph in entries[0] if ph[-1].isdigit())
    return _heuristic_syllables(word)


def compute_fkgl(text: str, sample_id: Optional[str] = None) -> FKGLResult:
    """
    FKGL formula:
        0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    Returns None for fkgl if text has zero words or zero sentences.
    """
    sentences = split_sentences(text)
    words = extract_words(text)

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words)

    if num_sentences == 0 or num_words == 0:
        return FKGLResult(
            id=sample_id,
            num_sentences=num_sentences,
            num_words=num_words,
            num_syllables=num_syllables,
            avg_words_per_sentence=0.0,
            avg_syllables_per_word=0.0,
            fkgl=None,
        )

    avg_words_per_sentence = num_words / num_sentences
    avg_syllables_per_word = num_syllables / num_words
    fkgl = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59

    return FKGLResult(
        id=sample_id,
        num_sentences=num_sentences,
        num_words=num_words,
        num_syllables=num_syllables,
        avg_words_per_sentence=avg_words_per_sentence,
        avg_syllables_per_word=avg_syllables_per_word,
        fkgl=fkgl,
    )


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def read_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_samples(input_path: Path, text_column: str) -> List[tuple[str, str]]:
    """
    Returns a list of (sample_id, text).
    Supports:
    - .txt file
    - directory of .txt files
    - .csv
    - .jsonl
    """
    if input_path.is_dir():
        samples: List[tuple[str, str]] = []
        for txt_file in sorted(input_path.glob("*.txt")):
            samples.append((txt_file.name, read_txt_file(txt_file)))
        return samples

    suffix = input_path.suffix.lower()

    if suffix == ".txt":
        return [(input_path.name, read_txt_file(input_path))]

    if suffix == ".csv":
        samples = []
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if text_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Text column '{text_column}' not found in CSV. "
                    f"Available columns: {reader.fieldnames}"
                )
            for i, row in enumerate(reader, start=1):
                sample_id = str(row.get("id", i))
                text = row.get(text_column, "")
                samples.append((sample_id, text))
        return samples

    if suffix == ".jsonl":
        samples = []
        with input_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = str(row.get("id", i))
                text = row.get(text_column, "")
                samples.append((sample_id, text))
        return samples

    raise ValueError(
        "Unsupported input format. Use a .txt file, a directory of .txt files, "
        "a .csv file, or a .jsonl file."
    )


def write_csv(results: List[FKGLResult], output_path: Path) -> None:
    fieldnames = [
        "id",
        "num_sentences",
        "num_words",
        "num_syllables",
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "fkgl",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow(row)


def write_jsonl(results: List[FKGLResult], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def print_single_result(result: FKGLResult) -> None:
    print(f"id: {result.id}")
    print(f"sentences: {result.num_sentences}")
    print(f"words: {result.num_words}")
    print(f"syllables: {result.num_syllables}")
    print(f"avg_words_per_sentence: {result.avg_words_per_sentence:.4f}")
    print(f"avg_syllables_per_word: {result.avg_syllables_per_word:.4f}")
    if result.fkgl is None:
        print("fkgl: None")
    else:
        print(f"fkgl: {result.fkgl:.4f}")


def print_summary(results: List[FKGLResult]) -> None:
    fkgl_mean = mean(r.fkgl for r in results)
    sent_mean = mean(r.num_sentences for r in results)
    word_mean = mean(r.num_words for r in results)
    syll_mean = mean(r.num_syllables for r in results)

    print(f"num_samples: {len(results)}")
    print(f"mean_sentences: {float(sent_mean) if sent_mean is not None else 'None'}")
    print(f"mean_words: {float(word_mean) if word_mean is not None else 'None'}")
    print(f"mean_syllables: {float(syll_mean) if syll_mean is not None else 'None'}")
    print(f"mean_fkgl: {fkgl_mean:.4f}" if fkgl_mean is not None else "mean_fkgl: None")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Flesch–Kincaid Grade Level (FKGL) for English text."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw input text.")
    group.add_argument(
        "--input",
        type=str,
        help="Path to a .txt file, a directory of .txt files, a .csv file, or a .jsonl file.",
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column/key name containing the text for CSV/JSONL input. Default: text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file (.csv or .jsonl) for batch results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample printing in batch mode and only show summary.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.text is not None:
        result = compute_fkgl(args.text, sample_id="input_text")
        print_single_result(result)
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    samples = load_samples(input_path, text_column=args.text_column)
    results = [compute_fkgl(text, sample_id=sample_id) for sample_id, text in samples]

    if not args.quiet:
        for result in results:
            print_single_result(result)
            print("-" * 40)

    print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_suffix = output_path.suffix.lower()

        if output_suffix == ".csv":
            write_csv(results, output_path)
        elif output_suffix == ".jsonl":
            write_jsonl(results, output_path)
        else:
            raise ValueError("Output file must end with .csv or .jsonl")

        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()