# Processes timestamped images generated by oakd_calibrate.py
# - Finds stereo captures within specified max dt and renames them
#   to the same filename

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    prog='Process timestamped data',
    description='Finds matching stereo images and ensure they have the same filename',
    epilog=''
)

parser.add_argument('left_path', type=Path, default=Path(Path.cwd() / 'left'),
                    help='Path to left images')
parser.add_argument('right_path', type=Path, default=Path(Path.cwd() / 'right'),
                    help='Path to right images')
parser.add_argument('max_dt_ms', type=float, default=100.0,
                    help='Max time difference between images')


def parse_timestamp(s: str) -> float:
    s = s.split(':')

    if len(s) != 3:
        return None

    hours, minutes, seconds = map(float, s)
    return hours*3600 + minutes*60 + seconds


if __name__ == "__main__":
    args = parser.parse_args()

    left = [(path, parse_timestamp(path.stem)) for path in args.left_path.glob('*.png')]
    right = [(path, parse_timestamp(path.stem)) for path in args.right_path.glob('*.png')]

    left.sort(key=lambda x: x[1])
    right.sort(key=lambda x: x[1])

    counter = 0
    for (r_path, r_stamp) in right:
        for (l_path, l_stamp) in left:
            if abs(r_stamp - l_stamp) < (args.max_dt_ms / 1000):
                l_path.rename(args.left_path / f"{counter}.png")
                r_path.rename(args.right_path / f"{counter}.png")
                counter += 1
                break

    print(f"Found {counter + 1} matching pairs.")
