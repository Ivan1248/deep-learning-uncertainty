import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage


def get_color_palette(n, cmap='jet'):
    cmap = mpl.cm.get_cmap(cmap)
    return [np.array(cmap(i / (n - 1))[:3]) for i in range(n)]


def fuse_images(im1, im2, a):
    return a * im1 + (1 - a) * im2


def colorify_label(lab, colors):
    plab = np.empty(list(lab.shape) + [3])
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            plab[i, j, :] = colors[lab[i, j]]
    return plab


def compose(images, format='0,0;1,0-1'):
    if format is None:
        return np.concatenate(
            [np.concatenate([im for im in row], 1) for row in images], 0)

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

    def __init__(self, name='Viewer'):
        self.name = name

    def display(self, dataset, mapping=lambda x: x):
        mpl.use('wxAgg')

        i = 0

        def on_press(event):
            nonlocal i
            if event.key == 'left':
                i -= 1
            elif event.key == 'right':
                i += 1
            elif event.key == 'q' or event.key == 'esc':
                plt.close(event.canvas.figure)
                return
            i = i % len(dataset)
            imgplot.set_data(mapping(dataset[i]))
            fig.canvas.set_window_title(str(i) + "-" + self.name)
            fig.canvas.draw()

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)
        fig.canvas.set_window_title(self.name)
        imgplot = ax.imshow(mapping(dataset[0]))
        plt.show()


def view_semantic_segmentation(dataset, infer=None):
    if False and 'class_colors' in dataset.info:
        colors = list(map(np.array, dataset.info['class_colors']))
        if np.max(np.array(colors)) > 1:
            colors = [c / 255 for c in colors]
    else:
        colors = get_color_palette(dataset.info['class_count'])
    colors = [np.zeros(3)] + list(map(np.array, colors))  # unknown black

    def get_frame(datapoint):
        img, lab = datapoint
        img_scaled01 = (img - np.min(img)) / (np.max(img) - np.min(img))
        clab = colorify_label(lab + 1, colors)
        cplab = clab if infer is None else colorify_label(
            infer(img) + 1, colors)

        print([(np.max(x), np.min(x)) for x in [img_scaled01, clab, cplab]])

        comp = compose([img_scaled01, clab, cplab], format='0,0;2,0-2;1,0-1')

        bar_width, bar_height = comp.shape[1] // 10, comp.shape[0]
        step = bar_height // len(colors)
        bar = np.zeros((bar_height, bar_width), dtype=np.int8)
        for i in range(len(colors)):
            bar[i * step:(i + 1) * step, 1:] = len(colors) - 1 - i
        bar = colorify_label(bar, colors)
        return compose([comp, bar], format='0,1')

    return Viewer().display(dataset, get_frame)