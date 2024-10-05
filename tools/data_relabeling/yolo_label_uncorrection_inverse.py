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
parser.add_argument("width")
parser.add_argument("height")

args = parser.parse_args()


def fix_labels(data_fpath, new_fpath, data_classes_fpath, width, height):
    data_classes_tmp = open(data_classes_fpath, "r").read().splitlines()
    data_classes = {}

    os.mkdir(new_fpath)
    os.mkdir(f"{new_fpath}/images")
    os.mkdir(f"{new_fpath}/labels")

    for i in range(len(data_classes_tmp)):
        data_classes[data_classes_tmp[i]] = i

    print("Data Classes:\n", data_classes)

    for file in os.listdir(data_fpath):
        if file.endswith(".txt"):
            print(file)
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
                    indv_data = obs.split(",")
                    indv_data_coords = indv_data[:len(indv_data) - 1]

                    x1, y1, x2, y2 = [float(x) for x in indv_data_coords]

                    data_class = data_classes[indv_data[len(indv_data) - 1]]

                    x_center = round(((x2 + x1) / 2) / width, 6)
                    y_center = round(((y2 + y1) / 2) / height, 6)

                    bb_width = round(abs(x2 - x1) / width, 6)
                    bb_height = round(abs(y2 - y1) / height, 6)

                    new_obs = [int(data_class), x_center, y_center, bb_width, bb_height]
                    print(f"new_obs:{new_obs}")
                    new_obss.append(new_obs)

                print(f"new_obss:{new_obss}")

                with codecs.open(f"{new_fpath}/labels/{file}", 'w', encoding='utf-8',
                                 errors='ignore') as new_fdata:
                    for obs in range(len(new_obss)):
                        f_line = ""
                        for elem in new_obss[obs]:
                            f_line += str(elem) + " "
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
    new_data_fpath = f"{args.input}-INVERSED"

    labels_fpath = args.input # + "/labels"
    fix_labels(labels_fpath, new_data_fpath, args.data_classes, float(args.width), float(args.height))

    images_fpath = args.input # + "/images"
    copy_images(images_fpath, new_data_fpath)
