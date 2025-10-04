import os
import csv


def split_tsv_by_rows(filepath, max_rows=100000, output_dir=None):
    """Split a TSV file by row count while preserving header and all columns.

    - filepath: path to the input .tsv file
    - max_rows: maximum number of data rows (excluding header) per output part
    - output_dir: directory to write parts into (defaults to same dir as input)

    The function will skip files that don't end with .tsv and will not re-split
    files that look like generated part files (ending with .partN.tsv).
    """
    if not filepath.lower().endswith('.tsv'):
        print(f"Skipping non-TSV file: {filepath}")
        return

    base = os.path.basename(filepath)
    # avoid re-processing already split files like name.part1.tsv
    if '.part' in base and base.rsplit('.part', 1)[1].split('.')[0].isdigit():
        print(f"Skipping generated part file: {filepath}")
        return

    if output_dir is None:
        output_dir = os.path.dirname(filepath)

    part_index = 1
    with open(filepath, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='\t')
        try:
            header = next(reader)
        except StopIteration:
            print(f"Empty TSV file: {filepath}")
            return

        rows_in_part = 0
        out_writer = None
        out_file = None

        def open_new_part(idx):
            name = f"{os.path.splitext(base)[0]}.part{idx}.tsv"
            path = os.path.join(output_dir, name)
            f = open(path, 'w', encoding='utf-8', newline='')
            w = csv.writer(f, delimiter='\t')
            w.writerow(header)
            print(f"Created part: {path}")
            return f, w

        out_file, out_writer = open_new_part(part_index)

        for row in reader:
            out_writer.writerow(row)
            rows_in_part += 1
            if rows_in_part >= max_rows:
                out_file.close()
                part_index += 1
                rows_in_part = 0
                out_file, out_writer = open_new_part(part_index)

        if out_file and not out_file.closed:
            out_file.close()


def run_on_ready_dir(ready_dir='./learning/ready', max_rows=100000):
    if not os.path.isdir(ready_dir):
        print(f"Ready directory not found: {ready_dir}")
        return

    for fname in os.listdir(ready_dir):
        fpath = os.path.join(ready_dir, fname)
        if os.path.isfile(fpath):
            try:
                split_tsv_by_rows(fpath, max_rows=max_rows, output_dir=ready_dir)
            except Exception as e:
                print(f"Failed to split {fpath}: {e}")


if __name__ == '__main__':
    # example default; callers can import and call split_tsv_by_rows directly
    run_on_ready_dir()