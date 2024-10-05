import random
import argparse
import glob
import pathlib
import re
from typing import Optional
import shutil

def get_ids(path: pathlib.Path) -> [str]:
    json_names = glob.glob("*.json", root_dir=path)

    ids = list(filter(lambda id: id is not None, [get_id(name) for name in json_names]))

    return ids


def get_id(name: str) -> Optional[str]:
    match = re.search(r"(\d+)\.json", name)
    if match:
        return match.group(1)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir")
    parser.add_argument("train_dir")
    parser.add_argument("train_split", type=float)
    parser.add_argument("val_dir")
    parser.add_argument("val_split", type=float)
    parser.add_argument("test_dir")
    parser.add_argument("test_split", type=float)

    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir).expanduser()
    train_dir = pathlib.Path(args.train_dir).expanduser()
    val_dir = pathlib.Path(args.val_dir).expanduser()
    test_dir = pathlib.Path(args.test_dir).expanduser()
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split

    if train_split + val_split + test_split > 1:
        print("bad splits")
        return

    ids = get_ids(in_dir)
    n_ids = len(ids)

    train_ids = random.sample(ids, round(train_split * n_ids))
    ids = list(filter(lambda id: id not in train_ids, ids))
    print(f"assigning {len(train_ids)} to train")

    val_ids = random.sample(ids, round(val_split * n_ids))
    ids = list(filter(lambda id: id not in val_ids, ids))
    print(f"assigning {len(val_ids)} to val")

    # test_ids = random.sample(ids, round(test_split * n_ids))
    # ids = list(filter(lambda id: id not in test_ids, ids))
    test_ids = ids
    print(f"assigning {len(test_ids)} to test")

    for id in train_ids:
        names = glob.glob(f"{id}*", root_dir=in_dir)

        for name in names:
            shutil.copy2(in_dir / name, train_dir)

    for id in val_ids:
        names = glob.glob(f"{id}*", root_dir=in_dir)

        for name in names:
            shutil.copy2(in_dir / name, val_dir)

    for id in test_ids:
        names = glob.glob(f"{id}*", root_dir=in_dir)

        for name in names:
            shutil.copy2(in_dir / name, test_dir)


if __name__ == "__main__":
    main()