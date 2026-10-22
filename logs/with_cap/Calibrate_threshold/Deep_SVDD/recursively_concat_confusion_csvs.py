import argparse
import csv
import os
import re
from typing import List
'''
Script to recursively concatenate confusion-matrix CSV files into one merged CSV.
The output CSV will have the same columns as the input files, sorted lexicographically.
Usage:
    python3 recursively_concat_confusion_csvs.py [input_folder] [-o output_csv] [--no-recursive]     
    - input_folder: Folder to search for CSV files (default: current folder)
    - output_csv: Path for the merged CSV (default: <input_folder>/all_confusions_merged.csv)
    - --no-recursive: Only include CSV files from the top-level folder (disable recursive scan)
Notes:
    - All input CSV files must have the same column names (order can differ).
    - The script will skip any CSV file that has a different header than the first one found.
    - The output CSV will contain all rows from the input files, with columns sorted alphabetically.
    
    - you can just run python3 recursively_concat_confusion_csvs.py
'''


def find_csv_files(root_path: str, recursive: bool) -> List[str]:
    """Find CSV files under root_path, excluding the output file if it already exists."""
    csv_files = []

    if recursive:
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    csv_files.append(os.path.join(dirpath, filename))
    else:
        for filename in os.listdir(root_path):
            full_path = os.path.join(root_path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(".csv"):
                csv_files.append(full_path)

    return sorted(csv_files)


def read_csv_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def read_csv_header(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV is empty: {csv_path}") from error


def normalize_header_name(name: str) -> str:
    """Normalize header token and drop accidental log/path suffixes.

    Example bad token:
        "sqlInjection2/home/.../test.py:53: FutureWarning: ..."
    becomes:
        "sqlInjection2"
    """
    if name is None:
        return ""

    token = str(name).strip().strip('"').strip("'")
    token = token.replace("\ufeff", "")
    token = " ".join(token.split())

    # Keep only the leading identifier-like prefix.
    # This safely removes appended absolute paths and warning strings.
    match = re.match(r"^[A-Za-z][A-Za-z0-9_]*", token)
    if match:
        return match.group(0)

    # Fallback: keep as-is if no identifier prefix exists.
    return token


def normalize_header(header: List[str]) -> List[str]:
    return [normalize_header_name(name) for name in header]


def build_lex_header(header: List[str]) -> List[str]:
    """Return case-insensitive lexicographically sorted header."""
    normalized = normalize_header(header)
    return sorted(normalized, key=lambda name: name.casefold())


def build_source_indices_for_target_header(header: List[str], target_header: List[str]) -> List[int]:
    """Return source indices that map a row from `header` into `target_header` order."""
    normalized = normalize_header(header)
    header_to_index = {name: index for index, name in enumerate(normalized)}
    return [header_to_index[name] for name in target_header]


def sort_csv_files_by_parent_label(csv_files: List[str], target_header: List[str]) -> List[str]:
    """Sort files by parent folder label according to target header order."""
    label_to_index = {name.casefold(): index for index, name in enumerate(target_header)}

    def sort_key(path: str):
        parent_label = os.path.basename(os.path.dirname(path)).casefold()
        if parent_label in label_to_index:
            return (0, label_to_index[parent_label], path.casefold())
        return (1, parent_label, path.casefold())

    return sorted(csv_files, key=sort_key)


def concatenate_csvs(input_folder: str, output_csv: str, recursive: bool):
    csv_files = find_csv_files(input_folder, recursive)
    output_abs = os.path.abspath(output_csv)

    csv_files = [path for path in csv_files if os.path.abspath(path) != output_abs]

    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_folder}")

    first_header = read_csv_header(csv_files[0])
    merged_header = build_lex_header(first_header)
    csv_files = sort_csv_files_by_parent_label(csv_files, merged_header)

    merged_rows = []

    for csv_path in csv_files:
        header, data_rows = read_csv_rows(csv_path)
        current_lex_header = build_lex_header(header)

        if current_lex_header != merged_header:
            raise ValueError(
                "Header mismatch found.\n"
                f"Expected: {merged_header}\n"
                f"Found in {csv_path}: {current_lex_header}"
            )

        source_indices = build_source_indices_for_target_header(header, merged_header)

        for row in data_rows:
            if len(row) != len(header):
                raise ValueError(
                    f"Row length mismatch in {csv_path}. "
                    f"Expected {len(header)} columns, got {len(row)}"
                )
            reordered_row = [row[index] for index in source_indices]
            merged_rows.append(reordered_row)

    output_dir = os.path.dirname(output_abs)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_abs, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(merged_header)
        writer.writerows(merged_rows)

    print(f"Merged {len(csv_files)} files into: {output_abs}")
    print(f"Total rows written (excluding header): {len(merged_rows)}")


def default_output_path(input_folder: str) -> str:
    return os.path.join(os.path.abspath(input_folder), "all_confusions_merged.csv")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate all CSV files in a folder into one CSV by appending rows. "
            "Input CSV files must contain the same column names; output columns are lexicographically sorted."
        )
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=".",
        help="Folder containing CSV files (default: current folder).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <input_folder>/all_confusions_merged.csv).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only include CSV files from the top-level folder (disable recursive scan).",
    )

    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    output_csv = args.output if args.output else default_output_path(input_folder)

    recursive = not args.no_recursive
    concatenate_csvs(input_folder=input_folder, output_csv=output_csv, recursive=recursive)


if __name__ == "__main__":
    main()
