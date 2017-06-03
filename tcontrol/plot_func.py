from matplotlib import pyplot as _plt

__all__ = ['grid', 'figure', 'show']


def show(*args, **kwargs):
    return _plt.show(*args, **kwargs)


def grid(b=None, which='major', axis='both', **kwargs):
    return _plt.grid(b, which, axis, **kwargs)


def figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True,
           FigureClass=_plt.Figure, **kwargs):
    return _plt.figure(num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, **kwargs)
