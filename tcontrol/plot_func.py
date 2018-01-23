from matplotlib import pyplot as _plt

__all__ = ['grid', 'figure', 'show']


def show(*args, **kwargs):
    return _plt.show(*args, **kwargs)


def grid(b=None, which='major', axis='both', **kwargs):
    return _plt.grid(b, which, axis, **kwargs)


def figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True,
           FigureClass=_plt.Figure, **kwargs):
    return _plt.figure(num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, **kwargs)


def _plot_response_curve(y, t, title):
    _plt.title(title)
    _plt.xlabel('t/s')
    _plt.ylabel('Amplitude')
    _plt.axvline(x=0, color='black')
    _plt.axhline(y=0, color='black')
    _plt.plot(t, y)
    _plt.grid()
    _plt.show()
