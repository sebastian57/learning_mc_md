from pathlib import Path
import re

import matplotlib.pyplot as plt


def _slugify(value, fallback="plot"):
    text = str(value).strip().lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def plot_xy(
    x,
    y,
    xlabel="x",
    ylabel="y",
    title=None,
    name="default",
    save=False,
    plots_dir="plots",
    show=True,
    ax=None,
    save_format="png",
    tight_layout=True,
    **plot_kwargs,
):
    """Plot x/y data with simple defaults and pass-through Matplotlib options.

    Parameters
    ----------
    x, y : array-like
        Data to plot.
    xlabel, ylabel : str
        Axis labels. Default to "x" and "y".
    title : str or None
        Plot title. If None, no title is set.
    name : str
        Subdirectory name under plots/ when save=True.
    save : bool
        If True, save the figure to plots/<name>/<title-with-spaces-replaced>.<ext>.
    plots_dir : str or Path
        Root directory for saved plots.
    show : bool
        If True, call plt.show().
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on. If None, create a new figure and axes.
    save_format : str
        File extension used when saving.
    tight_layout : bool
        If True, apply tight_layout() before showing/saving.
    **plot_kwargs
        Forwarded directly to ax.plot(...), e.g. color, linewidth, marker, alpha.

    Returns
    -------
    (fig, ax, save_path)
        save_path is None when save=False.
    """
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if x is None:
        ax.plot(y, **plot_kwargs)
    else:
        ax.plot(x, y, **plot_kwargs)

    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    if title is not None:
        ax.set_title(title)

    if tight_layout:
        fig.tight_layout()

    save_path = None
    if save:
        title_slug = _slugify(title if title else "plot")
        plot_dir = Path(plots_dir) / _slugify(name, fallback="default")
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / f"{title_slug}.{save_format.lstrip('.')}"
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, save_path


def plot_xy_multiple(
    plots,
    xlabel="x",
    ylabel="y",
    title=None,
    name="default",
    save=False,
    plots_dir="plots",
    show=True,
    ax=None,
    save_format="png",
    tight_layout=True,
    legend=False,
    legend_kwargs=None,
):
    """Plot multiple x/y series on the same axes.

    Parameters
    ----------
    plots : sequence of dict
        Each dict is passed to plot_xy(...) for one series. Typical keys are
        x, y, label, color, linestyle, linewidth, marker, alpha, etc.
    xlabel, ylabel, title, name, save, plots_dir, show, ax, save_format, tight_layout
        Same meaning as in plot_xy(...).
    legend : bool
        If True, draw a legend after all series are added.
    legend_kwargs : dict or None
        Extra keyword arguments passed to ax.legend(...).

    Returns
    -------
    (fig, ax, save_path)
        save_path is None when save=False.
    """
    if not plots:
        raise ValueError("plots must contain at least one series definition.")

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for plot_def in plots:
        series = dict(plot_def)
        x = series.pop("x", None)
        if "y" not in series:
            raise KeyError("Each plot definition must include a 'y' entry.")
        y = series.pop("y")

        plot_xy(
            x,
            y,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            ax=ax,
            show=False,
            save=False,
            tight_layout=False,
            **series,
        )

    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    if title is not None:
        ax.set_title(title)

    if legend:
        legend_kwargs = {} if legend_kwargs is None else dict(legend_kwargs)
        ax.legend(**legend_kwargs)

    if tight_layout:
        fig.tight_layout()

    save_path = None
    if save:
        title_slug = _slugify(title if title else "plot")
        plot_dir = Path(plots_dir) / _slugify(name, fallback="default")
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / f"{title_slug}.{save_format.lstrip('.')}"
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, save_path
