import argparse
import codecs
import os
import shutil

parser = argparse.ArgumentParser(
    description='label correction',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument("input", help="path to input folder containing videos")
parser.add_argument("data_classes", help="path to data_classes.txt file")
parser.add_argument("width", default="960")
parser.add_argument("height", default="540")

args = parser.parse_args()


def fix_labels(data_fpath, new_fpath, data_classes_fpath, width, height):
    data_classes_tmp = open(data_classes_fpath, "r").read().splitlines()
    data_classes = {}

    os.mkdir(new_fpath)
    os.mkdir(f"{new_fpath}/images")
    os.mkdir(f"{new_fpath}/labels")

    for i in range(len(data_classes_tmp)):
        data_classes[i] = data_classes_tmp[i]

    print("Data Classes:\n", data_classes)

    for file in os.listdir(data_fpath):
        if file.endswith(".txt"):
            print(f"file:{file}")
            with codecs.open(f"{data_fpath}/{file}", 'r', encoding='utf-8',
                             errors='ignore') as fdata:
                data = fdata.read().splitlines()
                data_clean = []

                for line in data:
                    data_clean.append(line)

            fdata.close()

            new_obss = []

            try:
                for obs in data_clean:
                    indv_data = obs.split(" ")
                    indv_data_coords = indv_data[1:]

                    x_center, y_center, bb_width, bb_height = [float(x) for x in indv_data_coords]
                    data_class = int(indv_data[0])

                    bb_width_px = bb_width * width
                    bb_height_px = bb_height * height

                    x1 = round((x_center * width) - (bb_width_px / 2), 1)
                    x2 = round((x_center * width) + (bb_width_px / 2), 1)
                    y1 = round((y_center * height) - (bb_height_px / 2), 1)
                    y2 = round((y_center * height) + (bb_height_px / 2), 1)

                    new_obs = [x1, y1, x2, y2, data_classes[data_class]]
                    print(f"new_obs:{new_obs}")
                    new_obss.append(new_obs)

                print(f"new_obss:{new_obss}")

                with codecs.open(f"{new_fpath}/labels/{file}", 'w', encoding='utf-8',
                                 errors='ignore') as new_fdata:
                    for obs in range(len(new_obss)):
                        f_line = ""
                        for (i, elem) in enumerate(new_obss[obs]):
                            f_line += str(elem)
                            if i != len(new_obss[obs]) - 1:
                                f_line += ","
                        if obs != len(new_obss) - 1:
                            new_fdata.write(f_line.strip() + "\n")
                        else:
                            new_fdata.write(f_line.strip())
            except ValueError as e:
                print(f"Error: {e} in file {file}! Continuing...")
            except KeyError as e:
                print(f"Error: {e}! No such class.")
            fdata.close()


def copy_images(data_fpath, new_fpath):
    for file in os.listdir(data_fpath):
        if file.endswith(".png"):
            shutil.copy(f"{data_fpath}/{file}", f"{new_fpath}/images")


if __name__ == "__main__":
    new_data_fpath = f"{args.input}-UNCORRECTED"
    fix_labels(args.input, new_data_fpath, args.data_classes, float(args.width), float(args.height))
    copy_images(args.input, new_data_fpath)
