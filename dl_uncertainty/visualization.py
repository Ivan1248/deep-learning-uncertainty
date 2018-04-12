import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_color_palette(n, black_first=True):
    cmap = mpl.cm.get_cmap('hsv')
    colors = [np.zeros(3)] if black_first else []
    colors += [
        np.array(cmap(0.8 * i / (n - len(colors) - 1))[:3])
        for i in range(n - len(colors))
    ]
    return colors


def fuse_images(im1, im2, a):
    return a * im1 + (1 - a) * im2


def compose(images, format='0,0;1,0-1'):
    if format is None:
        return np.concatenate([
            np.concatenate([im for im in row], 1) for row in images
        ], 0)

    def get_image(frc):
        inds = [int(i) for i in frc.split('-')]
        assert (len(inds) <= 2)
        ims = [images[i] for i in inds]
        return ims[0] if len(ims) == 1 else fuse_images(ims[0], ims[1], 0.5)

    format = format.split(';')
    format = [f.split(',') for f in format]
    return np.concatenate([
        np.concatenate([get_image(frc) for frc in frow], 1) for frow in format
    ], 0)


class Viewer:
    """
        Press "q" to close the window. Press anything else to change the displayed
        composite image. Press "a" to return to the previous image.
    """

    def __init__(self, name=None):
        self.name = name

    def display(self, data, mapping=lambda x: x):
        import matplotlib as mpl
        mpl.use('wxAgg')

        i = 0

        def on_press(event):
            nonlocal i
            if event.key == 'a':
                i -= 1
            elif event.key == 'q':
                plt.close(event.canvas.figure)
                return
            else:
                i += 1
            i = i % data.size
            imgplot.set_data(mapping(data[i]))
            fig.canvas.set_window_title(str(i) +"-" + self.name)
            fig.canvas.draw()

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)
        fig.canvas.set_window_title(self.name)
        imgplot = ax.imshow(mapping(data[0]))
        plt.show()