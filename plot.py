import sys

import numpy as np
from matplotlib import pyplot as plt


def f(losses, name):
    n_calls = len(losses)
    iterations = range(1, n_calls + 1)
    mins = [np.min(losses[:i]) for i in iterations]
    max_mins = max(mins)
    cliped_losses = np.clip(losses, None, max_mins)
    xlabel = "Number of iterations $n$"
    ylabel = r"Min objective value after $n$ iterations"
    ax = None
    alpha = 0.2
    yscale = None
    color = None
    true_minimum = None
    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)
    x = iterations
    y1 = mins
    y2 = cliped_losses
    ax.plot(x, y1, c=color, label=name)
    ax.scatter(x, y2, c=color, alpha=alpha)

    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum is not None or name is not None:
        ax.legend(loc="upper right")


if __name__ == '__main__':
    with open('scripts/objectives') as o:
        with open('scripts/acq_value') as a:
            o = o.readlines()
            a = a.readlines()
            o1 = []
            o2 = []
            for i in range(len(o)):
                e = eval(a[i].split()[2])
                if e == 0 or e == [0] or e == [[0]]:
                    continue
                j = eval(o[i])
                o1.append(j['Best'][0] if j['Best'] else sys.maxsize)
                o2.append(j['Second best'][0] if j['Second best'] else sys.maxsize)
            f(o1, 'Best')
            f(o2, 'Second best')
            plt.savefig('.png')
