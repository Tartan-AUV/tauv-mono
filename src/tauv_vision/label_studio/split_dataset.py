import glob
import pathlib
import argparse
import random
import shutil


def run(name: str, in_dir: pathlib.Path, out_root_dir: pathlib.Path, batch_size: int, zip: bool):
    paths = [path for path in in_dir.iterdir() if path.is_file()]

    batch_i = 0

    while True:
        if len(paths) == 0:
            break

        out_dir = out_root_dir / f"{name}_{batch_i}"
        out_dir.mkdir()

        print(f"Writing to {out_dir}...")

        if len(paths) < batch_size:
            print("Exhausted data!")

        selected_paths = random.sample(paths, batch_size)
        paths = list(filter(lambda path: path not in selected_paths, paths))

        for path in selected_paths:
            shutil.copy2(path, out_dir)

        print(f"Done writing to {out_dir}")

        if zip:
            out_zip = out_root_dir / f"{name}_{batch_i}"
            print(f"Zipping to {out_zip}...")
            shutil.make_archive(out_zip, "zip", root_dir=out_dir.parent, base_dir=f"{name}_{batch_i}")
            print(f"Done zipping to {out_zip}")

        batch_i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("in_dir")
    parser.add_argument("out_root_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--zip", action="store_true")

    args = parser.parse_args()

    in_dir = pathlib.Path(args.in_dir).expanduser()
    out_root_dir = pathlib.Path(args.out_root_dir).expanduser()

    assert in_dir.exists() and in_dir.is_dir()
    assert not out_root_dir.exists()

    out_root_dir.mkdir()

    run(args.name, in_dir, out_root_dir, args.batch_size, args.zip)


if __name__ == "__main__":
    main()
