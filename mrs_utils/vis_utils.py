import matplotlib.pyplot as plt


def get_default_colors():
    """
    Get plt default colors
    :return: a list of rgb colors
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def get_color_list():
    """
    Get default color list in plt, convert hex value to rgb tuple
    :return:
    """
    colors = get_default_colors()
    return [tuple(int(a.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for a in colors]


def compare_figures(images, nrows_ncols, show_axis=False, fig_size=(10, 8), show_fig=True):
    """
    Show three figures in a row, link their axes
    :param img_1: image to show on top left
    :param img_2: image to show at top right
    :param img_3: image to show on bottom left
    :param img_4: image to show on bottom right
    :param show_axis: if False, axes will be hide
    :param fig_size: size of the figure
    :param show_fig: show figure or not
    :param color_bar: if True, add color bar to the last plot
    :return:
    """
    from mpl_toolkits.axes_grid1 import Grid
    fig = plt.figure(figsize=fig_size)
    grid = Grid(fig, rect=111, nrows_ncols=nrows_ncols, axes_pad=0.25, label_mode='L', share_all=True)
    for i, (ax, img) in enumerate(zip(grid, images)):
        ax.imshow(img)
        if not show_axis:
            ax.axis('off')
    plt.tight_layout()
    if show_fig:
        plt.show()
