import numpy as np
from typing import Any, Final, List, Iterable, Optional, Tuple, Union

def rdShow(
    rda: "np.ndarray",
    ignore_colours=[],
    show: bool = True,
    axes: bool = True,
    cmap: str = "gray",
    figsize: Tuple[int, int] = (4, 4),
    zcolor: str = "red",
    zloc: int = 1,
):
    # if type(rda) is np.ndarray:
        # rda = rdarray(rda)
    if type(rda) is not np.ndarray:
        raise Exception("A numpy.ndarray is required!")

    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except:
        raise Exception("matplotlib must be installed to use rdShow!")
        try:
            # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from matplotlib.patches import Rectangle
        except:
            raise Exception("mpl_toolkits.axes_grid1 must be available!")

    # disparr = np.array(rda, copy=True)
    # disparr[disparr == rda.no_data] = np.nan
    # for c in ignore_colours:
    #     disparr[disparr == c] = np.nan
    vmin_calc, vmax_calc = np.nanpercentile(np.nan, [2, 98])
    if vmin is None:
        vmin = vmin_calc
    if vmax is None:
        vmax = vmax_calc

    fig, (ax, cax) = plt.subplots(
        ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [1, 0.05]}
    )

    # current_cmap = matplotlib.cm.get_cmap()
    # current_cmap.set_bad(color='red')
    iax = ax.imshow(disparr, vmin=vmin, vmax=vmax, cmap=cmap)

    fig.colorbar(iax, cax=cax)

    plt.tight_layout()

    if not axes:
        ax.axis("off")
    if show:
        plt.show()
    return 