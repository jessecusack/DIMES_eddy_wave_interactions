import numpy as np
import utils
from string import ascii_lowercase


def ylabel(label, fig, ax1, ax2, dx=None):
    x0 = 0.5*(ax1.get_position().x0 + ax2.get_position().x0)
    y0 = 0.5*(ax1.get_position().y0 + ax2.get_position().y1)
    if dx is None:
        dx = -x0
    x0 += dx
    fig.text(x0, y0, label, rotation='vertical', verticalalignment='center')


def xlabel(label, fig, ax1, ax2, dy=None):
    x0 = 0.5*(ax1.get_position().x0 + ax2.get_position().x1)
    y0 = 0.5*(ax1.get_position().y0 + ax2.get_position().y0)
    if dy is None:
        dy = -y0
    y0 += dy
    fig.text(x0, y0, label, rotation='horizontal',
             horizontalalignment='center')


def axes_labels(fig, axs, dx=0, dy=.01, i0=0, **kwargs):
    axs = np.asarray(axs)
    for i, ax in enumerate(axs.flat):
        bbox = ax.get_position()
        fig.text(bbox.x0+dx, bbox.y1+dy, '{})'.format(ascii_lowercase[i0+i]),
                 **kwargs)


def clump(x, nperseg):
    """Clump messy data."""
    idxs = utils.contiguous_regions(~np.isnan(x))
    ndats = np.squeeze(np.diff(idxs, axis=1))
    nsegs = np.floor(1.*ndats/nperseg).astype(int)
    nsegs_tot = np.sum(nsegs)
    xclumped = np.empty((nsegs_tot*nperseg, ))
    j = 0
    for row in idxs:
        i0, i1 = row
        ndat = i1 - i0
        if ndat < nperseg:
            continue
        nseg = int(np.floor(1.*ndat/nperseg))  # Convert to float and back.
        xseg = x[i0:i1]
        for i in range(nseg):
            xclumped[j*nperseg:(j+1)*nperseg] = xseg[i*nperseg:(i+1)*nperseg]
            j += 1
    return xclumped


def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    ** This function is copied directly from user Ralph on stackoverflow
    at https://stackoverflow.com/a/55067613 **
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='edge', width=widths,
           edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)


def nangmean(arr, axis=None):
    """Geometric mean that ignores NaNs."""
    arr = np.asarray(arr)
    valids = np.sum(~np.isnan(arr), axis=axis)
    prod = np.nanprod(arr, axis=axis)
    return np.power(prod, 1. / valids)
