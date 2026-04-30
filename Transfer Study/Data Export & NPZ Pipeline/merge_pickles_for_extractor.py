import os
import glob
import pickle
import argparse

def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        return obj
    return [obj]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder containing train/ valid/ test/")
    parser.add_argument("--pattern", action="append", required=True, help="Glob pattern, can be repeated")
    parser.add_argument("--output", required=True, help="Output merged .pkl file")
    args = parser.parse_args()

    files = []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(args.root, split)
        for pat in args.pattern:
            files.extend(glob.glob(os.path.join(split_dir, pat)))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No pickle files matched the given patterns")

    records = []
    for fp in files:
        records.extend(load_pickle(fp))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Matched files: {len(files)}")
    print(f"Merged records: {len(records)}")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
