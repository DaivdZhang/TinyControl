from matplotlib import pyplot as _plt
import matplotlib.widgets as widgets

__all__ = ['grid', 'figure', 'show', "AnnotatedPoint", "AttachedCursor"]


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


class AnnotatedPoint(object):
    def __init__(self, ax, fig):
        """

        :param ax:
        :type ax: matplotlib.axes.Axes
        :param fig:
        :type fig: matplotlib.figure.Figure
        """
        self.ax = ax
        self.fig = fig
        self.anno = self.init_annotate()
        self.dot = None
        self.cig = []

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("should be implemented in subclass.")

    def init_annotate(self):
        props = {"boxstyle": "square", "facecolor": "white", "alpha": 0.9}
        arrowprops = {"arrowstyle": '->', "connectionstyle": 'arc3, rad=0'}
        anno = self.ax.annotate('', xytext=(0.25, 0.25),
                                xy=(0, 0), bbox=props, arrowprops=arrowprops)

        return anno

    def connect(self, *events):
        for event in events:
            cig = self.fig.canvas.mpl_connect(event, self)
            self.cig.append(cig)

    def disconnect(self):
        for i in self.cig:
            self.fig.canvas.mpl_disconnect(i)


class AttachedCursor(widgets.Cursor):
    def __init__(self, ax, fig, **lineprops):
        super().__init__(ax, horizOn=True, vertOn=True, useblit=True,
                         **lineprops)
        self.ax = ax
        self.fig = fig
        self.fig = self.ax.figure
        self.cig = []
        self.ax.set_picker(True)

    def __call__(self, event):
        raise NotImplementedError("should be implemented in subclass.")

    def connect(self, *events):
        for event in events:
            cig = self.fig.canvas.mpl_connect(event, self)
            self.cig.append(cig)

    def disconnect(self):
        for i in self.cig:
            self.fig.canvas.mpl_disconnect(i)
