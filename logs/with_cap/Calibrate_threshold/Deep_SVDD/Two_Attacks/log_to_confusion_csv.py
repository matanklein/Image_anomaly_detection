import argparse
import csv
import os
import re


ATTACK_PREFIX = ">>> Setting up environment for:"


def normalize_confusion_row(row_line):
    """Normalize confusion row while preserving brackets."""
    normalized = row_line.strip()
    normalized = normalized.replace("[[", "[").replace("]]", "]")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("[ ", "[").replace(" ]", "]")
    return normalized


def normalize_to_pair(row_text):
    """Normalize any row representation to '[x y]' format."""
    values = re.findall(r"-?\d+", row_text)

    if len(values) >= 2:
        return f"[{values[0]} {values[1]}]"

    if len(values) == 1:
        return f"[0 {values[0]}]"

    return "[0 0]"


def extract_confusion_rows(lines, start_index):
    """Extract up to two confusion-matrix rows after 'Confusion Matrix:' line."""
    matrix_rows = []
    index = start_index + 1

    while index < len(lines):
        stripped = lines[index].strip()

        if not stripped:
            index += 1
            continue

        if (
            stripped.startswith("------------------------------")
            or stripped.startswith("TP (")
            or stripped.startswith("TN (")
            or stripped.startswith("FP (")
            or stripped.startswith("FN (")
            or stripped.startswith("FPR:")
            or stripped.startswith("Full Classification Report:")
            or stripped.startswith("Experiment ")
            or stripped.startswith(ATTACK_PREFIX)
        ):
            break

        if stripped.startswith("["):
            normalized = normalize_confusion_row(stripped)
            if normalized.startswith("[") and normalized.endswith("]"):
                matrix_rows.append(normalized)
                if len(matrix_rows) == 2:
                    break

        index += 1

    if not matrix_rows:
        return None

    if len(matrix_rows) == 1:
        single_row = normalize_to_pair(matrix_rows[0])
        if single_row == "[0 0]":
            return "[0 0]", "[0 0]"
        return "[0 0]", single_row

    return normalize_to_pair(matrix_rows[0]), normalize_to_pair(matrix_rows[1])


def parse_log_file(log_path):
    """Parse benchmark log and return a list of (attack_name, row1, row2)."""
    with open(log_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    experiments = []
    current_attack = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(ATTACK_PREFIX):
            current_attack = stripped.split(ATTACK_PREFIX, 1)[1].strip()
            continue

        if stripped == "Confusion Matrix:" and current_attack is not None:
            extracted = extract_confusion_rows(lines, index)
            if extracted is None:
                continue

            row1, row2 = extracted
            experiments.append((current_attack, row1, row2))

    return experiments


def write_csv(experiments, output_csv_path):
    """Write CSV where each attack is a column and two rows are matrix rows."""
    if not experiments:
        raise ValueError("No experiments with confusion matrices were found in the log.")

    headers = [attack for attack, _, _ in experiments]
    first_row = [row1 for _, row1, _ in experiments]
    second_row = [row2 for _, _, row2 in experiments]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(first_row)
        writer.writerow(second_row)


def default_output_path(input_log_path):
    base, _ = os.path.splitext(input_log_path)
    return f"{base}_confusion.csv"


def find_test_logs(root_path):
    """Recursively find all files named test.log under root_path."""
    test_logs = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename == "test.log":
                test_logs.append(os.path.join(dirpath, filename))
    return sorted(test_logs)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find all test.log files from the current path and export "
            "a confusion-matrix CSV for each one."
        )
    )
    parser.parse_args()

    root_path = os.getcwd()
    log_files = find_test_logs(root_path)

    if not log_files:
        print(f"No test.log files found under: {root_path}")
        return

    success_count = 0
    skipped_count = 0
    error_count = 0

    for log_path in log_files:
        output_path = default_output_path(log_path)
        try:
            experiments = parse_log_file(log_path)
            if not experiments:
                print(f"Skipped (no confusion-matrix experiments found): {log_path}")
                skipped_count += 1
                continue
            write_csv(experiments, output_path)
            print(f"Saved CSV with {len(experiments)} attack columns to: {output_path}")
            success_count += 1
        except Exception as error:
            print(f"Failed for {log_path}: {error}")
            error_count += 1

    print("\nDone.")
    print(f"CSV files created: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failures: {error_count}")


if __name__ == "__main__":
    main()