import logging
mpl_logger = logging.getLogger('matplotlib')  # must before import matplotlib
mpl_logger.setLevel(logging.WARNING)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pylab as plt


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = np.transpose(data, (2, 0, 1))
    return data


def plot_f0_to_numpy(f0_pre, f0_gt=None):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(f0_pre.T, "g")
    if f0_gt is not None:
        plt.plot(f0_gt.T, "r")
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
