#!/usr/bin/env python3
"""Save a random TEST-card selection, then write both datasets on acceptance.

Python 3.9+; no external packages. Functions may also be imported by notebooks.
Preview: python reduce_fraud_by_card.py preview --train fraudTrain.csv --test fraudTest.csv --percent 35
Accept:  python reduce_fraud_by_card.py write --plan sampling_plans/PLAN/selection.json
Replay:  python reduce_fraud_by_card.py reproduce --train fraudTrain.csv --test fraudTest.csv --cards evidence/existing_sample_cards.csv --audit evidence/sampling_audit.json --output-dir reproduced

Percent means unique TEST cards, rounded to the nearest whole card. Every
selected card keeps ALL transactions in both files. Each preview draws again
unless --seed is specified. Writing uses only the exact saved selection.
"""

# TEAM A - CARD-BASED DATA REDUCTION SUPPORTING PROGRAM
# Purpose: reduce Sparkov transaction files without cutting a selected card's
# history. Input files are the original fraudTrain.csv and fraudTest.csv.
#
# PHASE 1 (preview): --percent is a percentage of UNIQUE cards in TEST, not a
# requested percentage of transaction rows. Randomly select that many cards,
# save their exact IDs and seed, and count ALL their rows in BOTH input files.
# Print each resulting row percentage so the team can accept or reject the draw.
# Output: selection.json plus selected_cards.csv (IDs and per-file row counts).
# Preview does not write reduced transaction files. --seed repeats a draw;
# omitting --seed generates a fresh draw. --percent 35 is only an example.
#
# PHASE 2 (write): accept a saved selection.json, verify the original file
# fingerprints, and write ALL transactions for those saved cards from each
# file into a NEW output directory. Verify complete histories and row totals.
# Never truncate a history to reach a target percentage. Never overwrite input
# files or an existing output directory. Preserve the separate train/test roles.
#
# IMPORTANT HISTORICAL DISTINCTION: the report's existing 324-card subset used
# an earlier near-35%-of-TEST-ROWS selection rule. This revised unique-card
# percentage program will NOT regenerate those exact cards merely with 35%
# and seed 42. evidence/existing_sample_cards.csv records those exact historical
# IDs; evidence/sampling_audit.json verifies their row totals and source hashes.
# This program's newly generated selected_cards.csv is a DIFFERENT file for a
# new preview. The historical ID CSV is not a selection.json write-plan input.
# To replicate the REPORT'S EXACT SUBSET, use the reproduce command with that
# historical ID CSV and audit JSON. This command makes NO random selection:
# verify original source hashes, filter both originals by the saved IDs, check
# all retained histories/totals, and publish only to a NEW output directory.
#
# Card IDs are grouping keys retained as strings, not numerical risk features.
# Only card IDs/counts are used to select histories; fraud labels do not guide
# sampling. Selecting from TEST excludes cards that occur only in TRAIN.
# This program performs no EDA, feature engineering, model training or scoring.
# Python 3.9+ standard library only; also importable from notebooks. See the
# report's Appendix B for a plain-language description and Appendix A for the
# submission file guide. Full source datasets are not part of the submission.

import argparse
import csv
import hashlib
import json
import math
import random
import secrets
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_counts(path, column="cc_num", encoding="utf-8-sig"):
    counts = Counter()
    with Path(path).open(newline="", encoding=encoding) as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or column not in header or len(set(header)) != len(header):
            raise ValueError(f"{path}: missing card column or duplicate column names")
        index = header.index(column)
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            if len(row) != len(header) or not row[index].strip():
                raise ValueError(f"{path}: malformed row or empty card at record {line}")
            counts[row[index]] += 1
    if not counts:
        raise ValueError(f"{path}: no transactions")
    return counts


def preview_selection(train_path, test_path, percent, plan_dir=None, seed=None,
                      column="cc_num", encoding="utf-8-sig"):
    """Save one selection and counts, without creating reduced datasets."""
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("Percent must be greater than 0 and at most 100")
    train_path, test_path = Path(train_path).resolve(), Path(test_path).resolve()
    if train_path == test_path:
        raise ValueError("Train and test must be different files")
    seed = secrets.randbits(64) if seed is None else seed
    print("Counting all transactions by card...", flush=True)
    train = read_counts(train_path, column, encoding)
    test = read_counts(test_path, column, encoding)
    number = math.floor(len(test) * percent / 100 + 0.5)
    if number == 0:
        raise ValueError("This percentage rounds to zero cards; choose a larger percentage")
    # Sample cards, never individual transactions. Keep identifiers as strings.
    cards = sorted(random.Random(seed).sample(sorted(test), number))
    if plan_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        plan_dir = Path("sampling_plans") / f"{percent:g}pct_{stamp}_{seed}"
    plan_dir = Path(plan_dir).resolve()
    if plan_dir.exists():
        raise ValueError(f"Plan directory already exists; choose a new one: {plan_dir}")
    plan = {
        "format_version": 1,
        "method": "random_fraction_of_unique_test_cards",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_card_percent": percent, "seed": seed,
        "column": column, "encoding": encoding,
        "test_unique_cards": len(test), "selected_card_count": len(cards),
        "actual_test_card_percent": 100 * len(cards) / len(test),
        "selected_cards": cards, "sources": {}, "counts": {},
    }
    for name, path, counts in [("train", train_path, train), ("test", test_path, test)]:
        total = sum(counts.values())
        retained = sum(counts[card] for card in cards)
        plan["sources"][name] = {"path": str(path), "sha256": file_digest(path)}
        plan["counts"][name] = {
            "original_rows": total, "selected_rows": retained,
            "row_percent": 100 * retained / total,
            "selected_cards_present": len(set(cards) & counts.keys()),
        }
    plan_dir.mkdir(parents=True)
    with (plan_dir / "selected_cards.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([column, "train_transactions", "test_transactions", "total_transactions"])
        writer.writerows((card, train[card], test[card], train[card] + test[card]) for card in cards)
    plan_path = plan_dir / "selection.json"
    with plan_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    print(f"\nSelected {len(cards):,} of {len(test):,} unique test cards "
          f"({plan['actual_test_card_percent']:.2f}%). Seed: {seed}")
    print(f"{'File':<8} {'Original rows':>15} {'Selected rows':>15} {'Row percent':>13}")
    for name, counts in plan["counts"].items():
        print(f"{name:<8} {counts['original_rows']:>15,} {counts['selected_rows']:>15,} "
              f"{counts['row_percent']:>12.2f}%")
    print(f"\nSaved card selection and counts: {plan_path}")
    print("No reduced datasets created. To accept this selection, run:")
    print(f'python reduce_fraud_by_card.py write --plan "{plan_path}"')
    return plan_path


def write_selection(plan_path, output_dir=None, train_path=None, test_path=None):
    """Write complete histories from a saved plan; do not draw cards again."""
    plan_path = Path(plan_path).resolve()
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("format_version") != 1 or plan.get("method") not in {
        "random_fraction_of_unique_test_cards", "historical_card_ids_replay"
    }:
        raise ValueError("Unsupported sampling plan")
    card_list = plan["selected_cards"]
    if not card_list or not all(isinstance(card, str) and card.strip() for card in card_list):
        raise ValueError("Saved card selection is empty or invalid")
    cards = set(card_list)
    if len(cards) != len(card_list) or len(cards) != plan["selected_card_count"]:
        raise ValueError("Saved card selection contains duplicates or an incorrect count")
    paths = {
        "train": Path(train_path or plan["sources"]["train"]["path"]).resolve(),
        "test": Path(test_path or plan["sources"]["test"]["path"]).resolve(),
    }
    if paths["train"] == paths["test"]:
        raise ValueError("Train and test must be different files")
    output_dir = Path(output_dir).resolve() if output_dir else plan_path.parent / "reduced"
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; choose a new one: {output_dir}")
    print("Verifying that source files match the saved selection...", flush=True)
    for name, path in paths.items():
        if file_digest(path) != plan["sources"][name]["sha256"]:
            raise ValueError(f"The {name} input differs from the preview; create a new plan")

    # Publish only when both outputs match all per-card counts and preview totals.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sampling_", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "reduced"
        staging.mkdir()
        for name, path in paths.items():
            original_counts = read_counts(path, plan["column"], plan["encoding"])
            written = Counter()
            filename = "fraudTrain.csv" if name == "train" else "fraudTest.csv"
            with path.open(newline="", encoding=plan["encoding"]) as source, \
                    (staging / filename).open("x", newline="", encoding="utf-8") as target:
                reader, writer = csv.reader(source), csv.writer(target)
                header = next(reader)
                index = header.index(plan["column"])
                writer.writerow(header)
                for row in reader:
                    if row and row[index] in cards:
                        writer.writerow(row)
                        written[row[index]] += 1
            expected = Counter({card: original_counts[card] for card in cards if original_counts[card]})
            if written != expected or sum(written.values()) != plan["counts"][name]["selected_rows"]:
                raise ValueError(f"{name}: written histories do not match the saved preview")
            print(f"{filename}: {sum(written.values()):,} rows; complete histories verified")
        with (staging / "sampling_report.json").open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        staging.rename(output_dir)
    print(f"Saved reduced datasets: {output_dir}")
    return output_dir


def reproduce_sample(train_path, test_path, cards_path, audit_path, output_dir):
    """Replay exact historical card IDs, not the earlier randomized draw."""
    train_path, test_path, output_dir = map(lambda p: Path(p).resolve(), (train_path, test_path, output_dir))
    if train_path == test_path or output_dir.exists():
        raise ValueError('Inputs must differ and output directory must be NEW')
    with Path(cards_path).open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ['cc_num']:
            raise ValueError('Historical cards CSV must contain only the cc_num column')
        cards = [row['cc_num'] for row in reader]
    if not cards or any(not card or not card.strip() for card in cards) or len(set(cards)) != len(cards):
        raise ValueError('Historical card IDs must be nonempty and unique text values')
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    plan = {'format_version':1, 'method':'historical_card_ids_replay',
            'created_utc':datetime.now(timezone.utc).isoformat(), 'column':'cc_num',
            'encoding':'utf-8-sig', 'selected_cards':cards,
            'selected_card_count':len(cards), 'sources':{}, 'counts':{},
            'cards_sha256':file_digest(cards_path), 'audit_sha256':file_digest(audit_path)}
    for split, path in [('train',train_path), ('test',test_path)]:
        expected = audit['splits'][split]
        if file_digest(path) != expected['original_sha256']:
            raise ValueError(f'{split}: original source fingerprint differs from the historical audit')
        counts = read_counts(path)
        rows = sum(counts[card] for card in cards)
        if (sum(counts.values()) != expected['original_rows'] or rows != expected['sample_rows']
                or sum(card in counts for card in cards) != expected['sample_cards']
                or len(cards) != expected['sample_cards']):
            raise ValueError(f'{split}: saved cards do not match the historical audit counts')
        plan['sources'][split] = {'path':str(path), 'sha256':expected['original_sha256']}
        plan['counts'][split] = {'original_rows':sum(counts.values()), 'selected_rows':rows}
    # The normal checked writer preserves every selected row; no redraw occurs.
    with TemporaryDirectory(prefix='historical_card_replay_') as temporary:
        plan_path = Path(temporary) / 'selection.json'
        plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
        return write_selection(plan_path, output_dir, train_path, test_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    preview = phases.add_parser("preview", help="Save cards and report actual row percentages")
    preview.add_argument("--train", type=Path, required=True)
    preview.add_argument("--test", type=Path, required=True)
    preview.add_argument("--percent", type=float, required=True, help="Percentage of unique TEST cards")
    preview.add_argument("--seed", type=int, help="Optional fixed seed; omitted = a fresh draw")
    preview.add_argument("--plan-dir", type=Path, help="New folder for this selection")
    preview.add_argument("--column", default="cc_num")
    preview.add_argument("--encoding", default="utf-8-sig")
    write = phases.add_parser("write", help="Accept a saved plan and create both reduced CSVs")
    write.add_argument("--plan", type=Path, required=True)
    write.add_argument("--output-dir", type=Path, help="New output folder; default PLAN_FOLDER/reduced")
    write.add_argument("--train", type=Path, help="Source path override; content must match preview")
    write.add_argument("--test", type=Path, help="Source path override; content must match preview")
    replay = phases.add_parser('reproduce', help='Recreate the report subset using its exact historical card IDs')
    for name in ['train', 'test', 'cards', 'audit', 'output-dir']:
        replay.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "preview":
        preview_selection(args.train, args.test, args.percent, args.plan_dir, args.seed, args.column, args.encoding)
    elif args.phase == 'write':
        write_selection(args.plan, args.output_dir, args.train, args.test)
    else:
        reproduce_sample(args.train, args.test, args.cards, args.audit, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
