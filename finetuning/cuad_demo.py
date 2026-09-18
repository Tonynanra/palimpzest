import string

import numpy as np
import pandas as pd
try:
    from .cuad_data_loader import load_cuad_data
except ImportError:
    from cuad_data_loader import load_cuad_data

import palimpzest as pz

CUAD_CATEGORIES = [
    {
        "Category": "Document Name",
        "Description": "The name of the contract",
        "Answer Format": "Contract Name",
        "Group": "Group: -",
    },
    {
        "Category": "Parties",
        "Description": "The two or more parties who signed the contract",
        "Answer Format": "Entity or individual names",
        "Group": "Group: -",
    },
    {
        "Category": "Agreement Date",
        "Description": "The date of the contract",
        "Answer Format": "Date (mm/dd/yyyy)",
        "Group": "Group: 1",
    },
    {
        "Category": "Effective Date",
        "Description": "The date when the contract is effective\u00a0",
        "Answer Format": "Date (mm/dd/yyyy)",
        "Group": "Group: 1",
    },
    {
        "Category": "Expiration Date",
        "Description": "On what date will the contract's initial term expire?",
        "Answer Format": "Date (mm/dd/yyyy) / Perpetual",
        "Group": "Group: 1",
    },
    {
        "Category": "Renewal Term",
        "Description": "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.",
        "Answer Format": "[Successive] number of years/months / Perpetual",
        "Group": "Group: 1",
    },
    {
        "Category": "Notice Period to Terminate Renewal",
        "Description": "What is the notice period required to terminate renewal?",
        "Answer Format": "Number of days/months/year(s)",
        "Group": "Group: 1",
    },
    {
        "Category": "Governing Law",
        "Description": "Which state/country's law governs the interpretation of the contract?",
        "Answer Format": "Name of a US State / non-US Province, Country",
        "Group": "Group: -",
    },
    {
        "Category": "Most Favored Nation",
        "Description": "Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Non-Compete",
        "Description": "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?\u00a0",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "Exclusivity",
        "Description": "Is there an exclusive dealing\u00a0 commitment with the counterparty? This includes a commitment to procure all \u201crequirements\u201d from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on\u00a0 collaborating or working with other parties), whether during the contract or\u00a0 after the contract ends (or both).",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "No-Solicit of Customers",
        "Description": "Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "Competitive Restriction Exception",
        "Description": "This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.",
        "Answer Format": "Yes/No",
        "Group": "Group: 2",
    },
    {
        "Category": "No-Solicit of Employees",
        "Description": "Is there a restriction on a party\u2019s soliciting or hiring employees and/or contractors from the\u00a0 counterparty, whether during the contract or after the contract ends (or both)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Non-Disparagement",
        "Description": "Is there a requirement on a party not to disparage the counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Termination for Convenience",
        "Description": "Can a party terminate this\u00a0 contract without cause (solely by giving a notice and allowing a waiting\u00a0 period to expire)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Rofr/Rofo/Rofn",
        "Description": "Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Change of Control",
        "Description": "Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Anti-Assignment",
        "Description": "Is consent or notice required of a party if the contract is assigned to a third party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 3",
    },
    {
        "Category": "Revenue/Profit Sharing",
        "Description": "Is one party required to share revenue or profit with the counterparty for any technology, goods, or\u00a0services?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Price Restrictions",
        "Description": "Is there a restriction on the\u00a0 ability of a party to raise or reduce prices of technology, goods, or\u00a0 services provided?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Minimum Commitment",
        "Description": "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Volume Restriction",
        "Description": "Is there a fee increase or consent requirement, etc. if one party\u2019s use of the product/services exceeds certain threshold?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "IP Ownership Assignment",
        "Description": "Does intellectual property created\u00a0 by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Joint IP Ownership",
        "Description": "Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "License Grant",
        "Description": "Does the contract contain a license granted by one party to its counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Non-Transferable License",
        "Description": "Does the contract limit the ability of a party to transfer the license being granted to a third party?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Affiliate License-Licensor",
        "Description": "Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?\u00a0",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Affiliate License-Licensee",
        "Description": "Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Unlimited/All-You-Can-Eat-License",
        "Description": "Is there a clause granting one party an \u201centerprise,\u201d \u201call you can eat\u201d or unlimited usage license?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Irrevocable or Perpetual License",
        "Description": "Does the contract contain a\u00a0 license grant that is irrevocable or perpetual?",
        "Answer Format": "Yes/No",
        "Group": "Group: 4",
    },
    {
        "Category": "Source Code Escrow",
        "Description": "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,\u00a0 insolvency, etc.)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Post-Termination Services",
        "Description": "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Audit Rights",
        "Description": "Does a party have the right to\u00a0 audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: 5",
    },
    {
        "Category": "Uncapped Liability",
        "Description": "Is a party\u2019s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.",
        "Answer Format": "Yes/No",
        "Group": "Group: 6",
    },
    {
        "Category": "Cap on Liability",
        "Description": "Does the contract include a cap on liability upon the breach of a party\u2019s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        "Answer Format": "Yes/No",
        "Group": "Group: 6",
    },
    {
        "Category": "Liquidated Damages",
        "Description": "Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Warranty Duration",
        "Description": "What is the duration of any\u00a0 warranty against defects or errors in technology, products, or services\u00a0 provided under the contract?",
        "Answer Format": "Number of months or years",
        "Group": "Group: -",
    },
    {
        "Category": "Insurance",
        "Description": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Covenant Not to Sue",
        "Description": "Is a party restricted from contesting the validity of the counterparty\u2019s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
    {
        "Category": "Third Party Beneficiary",
        "Description": "Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?",
        "Answer Format": "Yes/No",
        "Group": "Group: -",
    },
]

DEFAULT_CUAD_CATEGORY_NAMES = [category["Category"] for category in CUAD_CATEGORIES]

# 0.15 is used in the Doc-ETL paper. It should be 0.5 for the actual benchmark.
IOU_THRESH = 0.15


def get_category_names() -> list[str]:
    return list(DEFAULT_CUAD_CATEGORY_NAMES)


def get_category_by_name(category_name: str) -> dict:
    categories_by_name = {category["Category"]: category for category in CUAD_CATEGORIES}
    try:
        return categories_by_name[category_name]
    except KeyError as exc:
        raise ValueError(f"Unknown CUAD category: {category_name}") from exc


def normalize_category_name(category_name: str) -> str:
    normalized = category_name.strip()
    normalized = normalized.replace(" For ", " for ")
    normalized = normalized.replace(" Of ", " of ")
    normalized = normalized.replace(" On ", " on ")
    normalized = normalized.replace(" Or ", " or ")
    normalized = normalized.replace(" To ", " to ")
    normalized = normalized.replace("Ip", "IP")
    return normalized


def resolve_selected_categories(
    selected_categories: list[str] | None = None,
    difficulty_buckets: dict[str, list[str]] | None = None,
) -> list[str]:
    if selected_categories is not None:
        resolved_categories = list(selected_categories)
    elif difficulty_buckets is not None:
        resolved_categories = []
        for difficulty in ("easy", "medium", "hard"):
            resolved_categories.extend(difficulty_buckets.get(difficulty, []))
    else:
        resolved_categories = get_category_names()

    validate_category_names(resolved_categories)
    return resolved_categories


def validate_category_names(category_names: list[str]) -> None:
    known_categories = set(get_category_names())
    duplicates = sorted({name for name in category_names if category_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate CUAD categories configured: {duplicates}")

    unknown_categories = [name for name in category_names if name not in known_categories]
    if unknown_categories:
        raise ValueError(f"Unknown CUAD categories configured: {unknown_categories}")


def _category_output_column(category: dict) -> dict:
    desc = (
        f"Extract the text spans (if they exist) from the contract corresponding to: {category['Description']}. "
        "If no spans exist, return an empty list. Quote text spans verbatim (do not summarize or paraphrase)."
    )
    return {"name": category["Category"], "type": list[str], "desc": desc}


def _extract_answer_texts(row: dict) -> list[str]:
    if isinstance(row["answers"], list):
        return [ans["text"] for ans in row["answers"]] if row["answers"] else []
    return row["answers"].get("text", [])


def get_label_df(
    num_contracts: int = 1,
    seed: int = 42,
    selected_categories: list[str] | None = None,
    split: str = "test",
    dataset_mode: str = "cuad-data",
    data_dir: str | None = None,
) -> pd.DataFrame:
    selected_categories = resolve_selected_categories(selected_categories)
    selected_category_set = set(selected_categories)
    dataset = load_cuad_data(split=split, data_dir=data_dir, dataset_mode=dataset_mode)

    # get the set of unique contract titles; to ensure the order of the contracts is
    # preserved, we use a list rather than using python's set()
    contract_titles = []
    for row in dataset:
        if row["title"] not in contract_titles:
            contract_titles.append(row["title"])

    # shuffle the contracts for the given seed
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(contract_titles)

    # get the first num_contracts
    contract_titles = contract_titles[:num_contracts]

    # construct the dataset one contract at a time
    final_label_dataset = []
    for title in contract_titles:
        # get the rows for this contract
        contract_rows = [row for row in dataset if row["title"] == title]

        # construct the contract; we get the contract_id and contract text from the first row
        contract = {
            "contract_id": contract_rows[0]["id"],
            "title": title,
            "contract": contract_rows[0]["context"],
        }
        if dataset_mode == "cuad-chunk":
            contract.update(
                {
                    "dataset_mode": dataset_mode,
                    "chunk_index": contract_rows[0].get("chunk_index"),
                    "chunk_start": contract_rows[0].get("chunk_start"),
                    "chunk_end": contract_rows[0].get("chunk_end"),
                    "source_paragraph_index": contract_rows[0].get("source_paragraph_index"),
                }
            )

        # add the labels
        contract.update({category_name: [] for category_name in selected_categories})
        for row in contract_rows:
            category_name = row["id"].split("__")[-1].split("_")[0].strip()
            category_name = normalize_category_name(category_name)
            if category_name not in selected_category_set:
                continue

            answer_texts = _extract_answer_texts(row)
            contract[category_name].extend(answer_texts)

        # add the contract to the dataset
        final_label_dataset.append(contract)

    return pd.DataFrame(final_label_dataset)


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

        # Convert the dataset into runnable rows. For cuad-chunk, each title is already a chunk-native unit.
        dataset = load_cuad_data(split=split, data_dir=data_dir, dataset_mode=dataset_mode)
        self.dataset = self._construct_dataset(dataset, num_contracts, seed)


    def _construct_dataset(self, dataset, num_contracts, seed: int=42):
        # get the set of unique contract titles; to ensure the order of the contracts is
        # preserved, we use a list rather than using python's set()
        contract_titles = []
        for row in dataset:
            if row["title"] not in contract_titles:
                contract_titles.append(row["title"])

        # shuffle the contracts for the given seed
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(contract_titles)

        # get the first num_contracts
        contract_titles = contract_titles[:num_contracts]

        # construct the dataset one contract at a time
        new_dataset = []
        for title in contract_titles:
            # get the rows for this contract
            contract_rows = [row for row in dataset if row["title"] == title]

            # construct the contract; we get the contract_id and contract text from the first row
            contract = {
                "contract_id": contract_rows[0]["id"],
                "title": title,
                "contract": contract_rows[0]["context"],
                "dataset_mode": self.dataset_mode,
                "chunk_index": contract_rows[0].get("chunk_index"),
            }

            # add the rows to the dataset
            new_dataset.append(contract)

        return new_dataset

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
