import string

import numpy as np
import pandas as pd

import palimpzest as pz

try:
    from .cuad_data import build_label_records, load_cuad_records, select_records
    from .cuad_data import load_cuad_data as load_cuad_data
    from .cuad_schema import CUAD_CATEGORIES as CUAD_CATEGORIES
    from .cuad_schema import DEFAULT_CUAD_CATEGORY_NAMES as DEFAULT_CUAD_CATEGORY_NAMES
    from .cuad_schema import _category_output_column, get_category_by_name, resolve_selected_categories
    from .cuad_schema import get_category_names as get_category_names
    from .cuad_schema import normalize_category_name as normalize_category_name
    from .cuad_schema import validate_category_names as validate_category_names
except ImportError:
    from cuad_data import build_label_records, load_cuad_records, select_records
    from cuad_data import load_cuad_data as load_cuad_data
    from cuad_schema import CUAD_CATEGORIES as CUAD_CATEGORIES
    from cuad_schema import DEFAULT_CUAD_CATEGORY_NAMES as DEFAULT_CUAD_CATEGORY_NAMES
    from cuad_schema import _category_output_column, get_category_by_name, resolve_selected_categories
    from cuad_schema import get_category_names as get_category_names
    from cuad_schema import normalize_category_name as normalize_category_name
    from cuad_schema import validate_category_names as validate_category_names

# 0.15 is used in the Doc-ETL paper. It should be 0.5 for the actual benchmark.
IOU_THRESH = 0.15


def get_label_df(
    num_contracts: int = 1,
    seed: int = 42,
    selected_categories: list[str] | None = None,
    split: str = "test",
    dataset_mode: str = "cuad-data",
    data_dir: str | None = None,
) -> pd.DataFrame:
    selected_categories = resolve_selected_categories(selected_categories)
    records = load_cuad_records(split=split, data_dir=data_dir, dataset_mode=dataset_mode)
    selected_records = select_records(records, num_contracts, seed)
    return pd.DataFrame(build_label_records(selected_records, selected_categories, dataset_mode))


#  Return the Jaccard similarity between two strings
def get_jaccard(label, pred):
    remove_tokens = [c for c in string.punctuation if c != "/"]
    for token in remove_tokens:
        label = label.replace(token, "")
        pred = pred.replace(token, "")
    label = label.lower()
    pred = pred.lower()
    label = label.replace("/", " ")
    pred = pred.replace("/", " ")

    label_words = set(label.split(" "))
    pred_words = set(pred.split(" "))

    intersection = label_words.intersection(pred_words)
    union = label_words.union(pred_words)
    jaccard = len(intersection) / len(union)
    return jaccard


# Find the number of true positives, false positives, and false negatives for each entry
# (one field extracted from each contract) by comparing the labels and predictions.
# Labels and preds are lists of strings
def evaluate_entry(labels, preds, substr_ok):
    tp, fp, fn = 0, 0, 0

    # jaccard similarity expects strings
    # TODO: This is a hack, ideally, the return type of the preds should be known
    for idx, pred in enumerate(preds):
        if not isinstance(pred, str):
            print(f"Expected string, but got {pred}")
            preds[idx] = str(pred)

    # first check if labels is empty
    if len(labels) == 0:
        if len(preds) > 0:
            fp += len(preds)  # false positive for each one
    else:
        for ans in labels:
            assert len(ans) > 0
            # check if there is a match
            match_found = False
            for pred in preds:
                if substr_ok:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                else:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH
                if is_match:
                    match_found = True

            if match_found:
                tp += 1
            else:
                fn += 1

        # now also get any fps by looping through preds
        for pred in preds:
            # Check if there's a match. if so, don't count (don't want to double count based on the above)
            # but if there's no match, then this is a false positive.
            # (Note: we get the true positives in the above loop instead of this loop so that we don't double count
            # multiple predictions that are matched with the same answer.)
            match_found = False
            for ans in labels:
                assert len(ans) > 0
                if substr_ok:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH or ans in pred
                else:
                    is_match = get_jaccard(ans, pred) >= IOU_THRESH
                if is_match:
                    match_found = True

            if not match_found:
                fp += 1

    return tp, fp, fn


def handle_empty_preds(preds):
    if preds is None or (  # noqa: SIM114
        isinstance(preds, str) and (preds == "" or preds == " " or preds == "null" or preds == "None")
    ):
        return []
    elif isinstance(preds, float) and np.isnan(preds):
        return []
    if not isinstance(preds, (list, np.ndarray)):
        return [preds]
    return preds


class CUADDataset(pz.IterDataset):
    def __init__(
        self,
        num_contracts: int = 1,
        split: str = "train",
        seed: int = 42,
        dataset_mode: str = "cuad-data",
        data_dir: str | None = None,
    ):
        self.num_contracts = num_contracts
        self.split = split
        self.seed = seed
        self.dataset_mode = dataset_mode
        self.data_dir = data_dir

        input_cols = [
            {"name": "contract_id", "type": str, "desc": "The id of the the contract to be analyzed"},
            {"name": "title", "type": str, "desc": "The title of the the contract to be analyzed"},
            {"name": "contract", "type": str, "desc": "The content of the the contract to be analyzed"},
            {"name": "dataset_mode", "type": str, "desc": "The CUAD dataset mode used for this benchmark row"},
            {"name": "chunk_index", "type": int, "desc": "The chunk index for chunk-native CUAD rows"},
        ]
        super().__init__(id="cuad", schema=input_cols)

        records = load_cuad_records(split=split, data_dir=data_dir, dataset_mode=dataset_mode)
        self.dataset = self._construct_dataset(records, num_contracts, seed)

    def _construct_dataset(self, records, num_contracts, seed: int = 42):
        return [
            {
                "contract_id": record["contract_id"],
                "title": record["title"],
                "contract": record["contract"],
                "dataset_mode": self.dataset_mode,
                "chunk_index": record.get("chunk_index"),
            }
            for record in select_records(records, num_contracts, seed)
        ]

    def __len__(self):
        return self.num_contracts

    def __getitem__(self, idx: int):
        return self.dataset[idx]

# Compute the precision and recall for the entire dataset.
# Each row in the dataframes should correspond to a contract.
# The columns should be the extracted fields (categories in CUAD_CATEGORIES).
def labels_to_long(
    label_df: pd.DataFrame,
    selected_categories: list[str] | None = None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    selected_categories = resolve_selected_categories(selected_categories)
    metadata = metadata or {}
    rows = []
    for _, label_row in label_df.iterrows():
        for category in selected_categories:
            row = {
                **metadata,
                "contract_id": label_row["contract_id"],
                "title": label_row.get("title"),
                "category": category,
                "labels": label_row.get(category, []),
            }
            for field in ("dataset_mode", "chunk_index", "chunk_start", "chunk_end", "source_paragraph_index"):
                if field in label_row:
                    row[field] = label_row.get(field)
            rows.append(row)
    return pd.DataFrame(rows)


def normalize_predictions_to_long(
    preds_df: pd.DataFrame,
    selected_categories: list[str] | None = None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    selected_categories = resolve_selected_categories(selected_categories)
    metadata = metadata or {}
    rows = []
    for _, pred_row in preds_df.iterrows():
        for category in selected_categories:
            row = {
                **metadata,
                "contract_id": pred_row["contract_id"],
                "title": pred_row.get("title"),
                "category": category,
                "predictions": handle_empty_preds(pred_row[category]) if category in pred_row else [],
            }
            for field in ("dataset_mode", "chunk_index", "chunk_start", "chunk_end", "source_paragraph_index"):
                if field in pred_row:
                    row[field] = pred_row.get(field)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics(
    label_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    selected_categories: list[str] | None = None,
) -> dict:
    selected_categories = resolve_selected_categories(selected_categories)
    label_long = labels_to_long(label_df, selected_categories)
    pred_long = normalize_predictions_to_long(preds_df, selected_categories)
    merged = label_long.merge(
        pred_long[["contract_id", "category", "predictions"]],
        on=["contract_id", "category"],
        how="left",
    )

    aggregate_tp, aggregate_fp, aggregate_fn = 0, 0, 0
    per_category = {}
    for category in selected_categories:
        category_rows = merged[merged["category"] == category]
        category_tp, category_fp, category_fn = 0, 0, 0
        substr_ok = "Parties" in category

        for _, row in category_rows.iterrows():
            labels = row["labels"]
            assert isinstance(labels, list)
            preds = handle_empty_preds(row["predictions"])
            entry_tp, entry_fp, entry_fn = evaluate_entry(labels, preds, substr_ok)
            category_tp += entry_tp
            category_fp += entry_fp
            category_fn += entry_fn

        precision = category_tp / (category_tp + category_fp) if category_tp + category_fp > 0 else np.nan
        recall = category_tp / (category_tp + category_fn) if category_tp + category_fn > 0 else np.nan
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        per_category[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": category_tp,
            "fp": category_fp,
            "fn": category_fn,
        }
        aggregate_tp += category_tp
        aggregate_fp += category_fp
        aggregate_fn += category_fn

    precision = aggregate_tp / (aggregate_tp + aggregate_fp) if aggregate_tp + aggregate_fp > 0 else np.nan
    recall = aggregate_tp / (aggregate_tp + aggregate_fn) if aggregate_tp + aggregate_fn > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": aggregate_tp,
        "fp": aggregate_fp,
        "fn": aggregate_fn,
        "per_category": per_category,
    }


def compute_precision_recall(label_df, preds_df, selected_categories: list[str] | None = None):
    tp, fp, fn = 0, 0, 0
    selected_categories = resolve_selected_categories(selected_categories)

    label_df = label_df.sort_values("contract_id").reset_index(drop=True)
    preds_df = preds_df.sort_values("contract_id").reset_index(drop=True)

    assert len(label_df) == len(preds_df), (
        f"Label and prediction dataframes have different row counts, label rows: {len(label_df)} vs preds rows {len(preds_df)}"
    )

    for label_row, pred_row in zip(label_df.iterrows(), preds_df.iterrows()):
        assert label_row[1]["contract_id"] == pred_row[1]["contract_id"], (
            f"IDs do not match. label id: {label_row[1]['contract_id']} vs pred id: {pred_row[1]['contract_id']}"
        )
        for category in selected_categories:
            substr_ok = "Parties" in category

            labels = label_row[1][category]
            assert isinstance(labels, list)

            preds = pred_row[1].get(category, [])
            preds = handle_empty_preds(preds)

            entry_tp, entry_fp, entry_fn = evaluate_entry(labels, preds, substr_ok)
            tp += entry_tp
            fp += entry_fp
            fn += entry_fn

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    return precision, recall


def build_cuad_query(dataset, mode, selected_categories: list[str] | None = None):
    assert mode in ["one-convert", "separate-converts", "bonded", "sequential"]
    selected_categories = resolve_selected_categories(selected_categories)
    categories = [get_category_by_name(category_name) for category_name in selected_categories]

    if mode in ["one-convert", "bonded"]:
        cols = [_category_output_column(category) for category in categories]
        dataset = dataset.sem_map(cols, depends_on=["contract"])
    elif mode in ["separate-converts", "sequential"]:
        for category in categories:
            dataset = dataset.sem_map(
                [_category_output_column(category)],
                depends_on=["contract"],
            )

    return dataset
