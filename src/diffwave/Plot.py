import random

import numpy as np
import torch
from matplotlib import pyplot as plt, patches


def newfig(n, w=32, h=32):
    fign = plt.figure(n)
    fign.set_figwidth(w)
    fign.set_figheight(h)

    return fign

def get_figidx(ax):
    fig = ax.get_figure()
    return [i for i in plt.get_fignums() if plt.figure(i) == fig][0]


def plotCont1d(ax, data, title, xlim=None, ylim=None):
    print("datatensor =", data)
    ax.plot(range(0, len(data)), data)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)

    fig = ax.get_figure()
    fig.savefig(str(get_figidx(ax)) + '.svg')

def plotHeatmap2d(ax, data: torch.Tensor, title, showvals = False, filetype='svg'):

    im = ax.imshow(data)

    fig = ax.get_figure()
    textsize = fig.get_size_inches()[0] * 6 / 32
    # Loop over data dimensions and create text annotations.
    for iy, row in enumerate(data):
        for ix, x in enumerate(row):
            #iy = i // data.shape[1]
            #ix = i - iy * data.shape[0]
            #print(iy, ix, "    |   ", x)
            if showvals:
                text = ax.text(ix, iy, round(data[iy, ix].item(), 3), ha="center", va="center", color="w", fontsize=textsize)

    ax.set_title(title)

    fig.savefig(str(get_figidx(ax)) + '.' + filetype)
    #fig.savefig('createwety.svg')