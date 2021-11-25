import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import os

import math


def split_dict(profile, stepKey, profileList, labelList, discardKeys):
    newDict = {}
    for key in profile:
        if type(profile[key]) is dict:
            newDict[key] = profile[key]['total']
            profileList, labelList = split_dict(profile[key],
                                                key,
                                                profileList,
                                                labelList,
                                                discardKeys)

        else:
            if key not in discardKeys:
                newDict[key] = profile[key]
    profileList.append(newDict)
    labelList.append(stepKey)
    return profileList, labelList


def mean_profile(profileList, nbCalls):
    for entry in profileList:
        for key in entry:
            entry[key] /= nbCalls
    return profileList


def plot_profile(filename, savefile, multi=False):
    if multi:
        nb_plots = len(filename)
        print(nb_plots)
        width = 2
        height = int(math.ceil(nb_plots/2))
        fig, ax = plt.subplots(width, height, figsize=(15, 15))
        for i in range(width):
            for j in range(height):
                discardKeys = ['calls', 'horizon']
                with open(filename[i*width+height], 'r') as stream:
                    profile = yaml.safe_load(stream)
                profileList, labels = split_dict(profile,
                                                 'step',
                                                 [],
                                                 [],
                                                 discardKeys)

                nb_calls = profile['calls']
                avgProfileList = mean_profile(profileList, nb_calls)

                plot_cumulative(ax[i, j], avgProfileList, labels)
    else:
        fig, ax = plt.subplots(figsize=(15, 15))
        discardKeys = ['calls', 'horizon']
        with open(filename, 'r') as stream:
            profile = yaml.safe_load(stream)
        profileList, labels = split_dict(profile, 'step', [], [], discardKeys)

        nb_calls = profile['calls']
        avgProfileList = mean_profile(profileList, nb_calls)

        plot_cumulative(ax, avgProfileList, labels)

    plt.savefig(savefile)


def plot_cumulative(ax, meanTimes, labels):
    '''
        Plots cummulative barplot for the routnies profiled.
        inputs:
        -------
            - meanTimes: list, every entry is one dictionnary summary
                at a given step.
            - labels: list of lenght depth, containing the labels for
                the different subroutines parsed.
            - bar_labels: a list of len depth where every entry is a list
                containing the labels for every bar.
    '''
    width = 0.35       # the width of the bars: can also be len(x) sequence

    entries = len(meanTimes)
    div = np.zeros(shape=(entries))
    totPlot = np.zeros(shape=(entries))

    for i, entry in enumerate(meanTimes):
        toPlot, barLabels, div = prep_values(entries, i, entry, div)
        totPlot = add_bottom(toPlot, totPlot)
        plot_bar(ax, labels, toPlot, barLabels, width)

    plot_div(ax, labels, div, width, totPlot)

    ax.set_ylabel('Time (s)')
    ax.set_title('MPPI profiling')
    ax.legend(bbox_to_anchor=(1.1, 1.05))


def prep_values(entries, idx, times, div):
    len_times = len(times)-1
    values = np.zeros(shape=(len_times, entries))
    sorted = np.zeros(shape=(len_times, entries))
    bar_labels = []
    sorted_labels = []
    # hack
    foo = 0
    for j, key in enumerate(times):
        if key != "total":
            values[j-foo, idx] = times[key]
            bar_labels.append(key)
        else:
            foo = 1

    div[idx] = times['total'] - np.sum(values)
    indexes = np.argsort(values, axis=0, kind='quicksort')[:, idx]

    for i in range(len_times):
        sorted[i, idx] = values[indexes[len_times - i - 1], idx]
        sorted_labels.append(bar_labels[indexes[len_times - i - 1]])

    return sorted, sorted_labels, div


def add_bottom(toPlot, totPlot):
    s = np.sum(toPlot, axis=0)
    totPlot += s
    return totPlot


def plot_div(ax, labels, div, width, bottom):
    ax.bar(labels,
           div,
           width,
           bottom=bottom,
           label='diverse',
           color='black')
    pass


def plot_bar(ax, labels, values, barLabels, width):
    '''
        Plots a bar cummulative bar plot.
        inputs:
        -------
            - ax, a matplotlib figure axes.
            - labels, list of strings, the labels
                for the different barplots.
            - values, numpy array containing the values to plot.
            - bar_labels, list of string labeling the differents bars.
            - width, the width of the bars.
    '''
    combined = 0
    for i, value in enumerate(values):
        ax.bar(labels, value, width, bottom=combined, label=barLabels[i])
        combined += value


if __name__ == "__main__":
    # Does not currently have support to read files from folders recursively
    parser = ap.ArgumentParser(description='Reads a single or a set \
                                                  of profiling files, and \
                                                  saves the results with \
                                                  barplots.',
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path',
                        nargs='+',
                        help='Path of a file or a folder of files.')

    parser.add_argument('-d',
                        '--dir',
                        default='',
                        help="Path to the saving directory")

    parser.add_argument('-m',
                        '--multi',
                        action='store_true',
                        help='if set to true, this plots all the files \
                             on the same image.')

    parser.add_argument('-s',
                        '--save',
                        default='',
                        help='Path to save file. When flag multi is true, \
                             then the image is saved in this file. Else \
                             it has no effect. ')

    args = parser.parse_args()

    # Parse paths

    fullPaths = [os.path.join(os.getcwd(), path) for path in args.path]

    files = []
    saveFiles = []
    for path in fullPaths:
        if os.path.isfile(path):
            files.append(path)
            file = os.path.basename(path)
            filename = os.path.splitext(file)[0]
            savePath = os.path.join(os.getcwd(), args.dir, filename + '.png')
            saveFiles.append(savePath)
        else:
            raise "One of the files doesn't exist"

    if args.multi:
        s = os.path.join(os.getcwd(), args.dir, args.save)
        if os.path.isdir(s):
            raise "Error, encountered a dir as savefile when \
                  multi is activated"
        plot_profile(files, s, True)

    else:
        for f, s in zip(files, saveFiles):
            plot_profile(f, s)
