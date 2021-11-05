import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jetson stats visualizer")
    parser.add_argument("--file", action="store", dest="file")

    choices = ["CPU", "GPU", "RAM"]
    parser.add_argument("data", action="store_true", choices=choices)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"{args.file} does not exist")

    df = pd.read_csv(args.file)

    if args.data == "CPU":
        df["CPU"] = int((df["CPU1"] + df["CPU2"] + df["CPU3"] + df["CPU4"] + df["CPU5"] + df["CPU6"] + df["CPU7"] + df["CPU8"]) / 8)

    x = df['uptime']
    y1 = df[f'{args.choice}']
    y2 = df[f'Temp {args.choice}']

    fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
    ax1.plot(x, y1, color='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color='tab:blue')

    ax1.set_xlabel('Usage', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel(f'{args.choice}', color='tab:red', fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
    ax1.grid(alpha=.4)

    ax2.set_ylabel("Temp", color='tab:blue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), 60))
    ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
    ax2.set_title(f"{args.choice}", fontsize=22)
    fig.tight_layout()
    plt.show()