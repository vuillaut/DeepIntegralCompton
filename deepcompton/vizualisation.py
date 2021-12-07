import matplotlib.pyplot as plt
import numpy as np


def hist2d_plans(data, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].set_title('ISGRI')
    _, _, _, cm = axes[0].hist2d(data.y1, data.z1, **kwargs)
    fig.colorbar(cm, ax=axes[0])
    axes[1].set_title('PICSIT')
    _, _, _, cm = axes[1].hist2d(data.y2, data.z2, **kwargs)
    fig.colorbar(cm, ax=axes[1])

    for ax in axes.ravel():
        ax.set_xlabel('y / cm')
        ax.set_ylabel('z / cm')

    return axes


def scatter_plans(data, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title('ISGRI')
    cm = axes[0].scatter(data.y1, data.z1, c=data.e1, **kwargs)
    cbar = fig.colorbar(cm, ax=axes[0])
    cbar.set_label('e1 / KeV')

    axes[1].set_title('PICSIT')
    cm = axes[1].scatter(data.y2, data.z2, c=data.e2, **kwargs)
    cbar = fig.colorbar(cm, ax=axes[1])
    cbar.set_label('e2 / KeV')

    for ax in axes.ravel():
        ax.set_xlabel('y / cm')
        ax.set_ylabel('z / cm')

    return axes


def plot_backprojected(theta_g, r_g, density, figsize=(12, 9)):
    """
    plot backprojected map in polar

    :param theta_g:
    :param r_g:
    :param density:
    :param figsize:
    :return:
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    ax.grid(False)
    cax = ax.pcolormesh(theta_g, r_g, density, cmap="hot")
    cbar = fig.colorbar(cax, label="Number of intersecting cones")

    tt = ax.get_yticklabels()
    list_tt = np.linspace(90/np.size(tt), 90, np.size(tt))
    for i in range(np.size(tt)):
        tt[i].set_text(str(int(list_tt[i]))+"°")
        tt[i].set_color("grey")
        tt[i].set_fontweight(900)
    ax.set_yticklabels(tt)
    ax.legend()
    ax.grid(True)
    return ax


def plot_source_pos(theta_source, phi_source, ax=None):
    """
    plot source position in polar

    :param theta_source:
    :param phi_source:
    :param ax:
    :return:
    """
    ax = plt.gca() if ax is None else ax
    ax.scatter(phi_source, theta_source, label="Source position")
    ax.legend()
    return ax

