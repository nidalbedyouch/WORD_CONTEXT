# A script to plot tsne data
# using matplotlib
import getopt
import sys
from pathlib import Path
from str2bool import str2bool
import matplotlib.pyplot as plt


def tsne_plot(input_filename, skip_header):
    # load 2D tsne data
    print('loading tsne data...')
    labels = []
    points = []
    header = None
    with open(str(input_filename)) as f:
        for i, line in enumerate(f):
            if i == 0 and skip_header:
                header = line
                continue
            data = line.split(',')
            syllable = data[0]
            x = float(data[1])
            y = float(data[2])
            labels.append(syllable)
            points.append([x, y])
    print('- found ' + str(len(labels)) + ' unique words')

    x_coords = []
    y_coords = []
    for point in points:
        x_coords.append(point[0])
        y_coords.append(point[1])
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def main(argv):
    try:
        options, args = getopt.getopt(argv, 'f:s:', ['file_path=', 'skip_header'])
        if len(options) == 0 or len(options) > 2:
            print('usage : tsne_plot.py -f <file_path> -s <skip_header>')
            sys.exit(2)
        else:
            skip_header = False
            for flag, arg in options:
                if '-f' in flag or '--file_path' in flag:
                    file_path = Path(arg)
                    if not file_path.exists():
                        assert False, 'file doesn\'t exist'
                elif '-s' in flag or '--skip_header' in flag:
                    skip_header = str2bool(arg)
                else:
                    assert False, 'unexpected argument'
            tsne_plot(file_path, skip_header)
    except getopt.GetoptError:
        print('usage : tsne_plot.py -f <file_path> -s <skip_header>')
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
