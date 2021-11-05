from jtop import jtop, JtopException
import csv
import argparse
import os
from datetime import datetime


if __name__ == "__main__":
    now = datetime.now()
    parser = argparse.ArgumentParser(description="jetson profiler")
    parser.add_argument("--file", action="store", dest="file", default=f"{now.strftime('%Y-%H-%M-%S')}.csv")
    args = parser.parse_args()

    if not os.path.exists('logs'):
        os.makedirs('logs')

    file = f"logs/{args.file}"

    print(f"Saving log on {file}")

    try:
        with jtop() as jetson:
            with open(file, "w") as csvfile:
                stats = jetson.stats

                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()
                writer.writerow(stats)

                while jetson.ok():
                    stats = jetson.stats
                    writer.writerow(stats)
                    print(f"Log at {stats['time']}")
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")