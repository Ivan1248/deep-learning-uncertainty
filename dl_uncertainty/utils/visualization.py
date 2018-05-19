import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage
from functools import lru_cache


def get_color_palette(n, cmap='jet'):
    cmap = mpl.cm.get_cmap(cmap)
    return [np.array(cmap(i / (n - 1))[:3]) for i in range(n)]


def fuse_images(im1, im2, a=0.5):
    return a * im1 + (1 - a) * im2


def colorify_label(lab, colors):
    plab = np.empty(list(lab.shape) + [3])
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            plab[i, j, :] = colors[lab[i, j]]
    return plab


def compose(images_in_array):
    if type(images_in_array[0]) is not list:
        images_in_array = [images_in_array]
    rows = [np.concatenate(row, axis=1) for row in images_in_array]
    return np.concatenate(rows, axis=0)


def compose_old(images, format='0,0;1,0-1'):
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
        #mpl.use('wxAgg')

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
    if 'class_colors' in dataset.info:
        colors = list(map(np.array, dataset.info['class_colors']))
        if np.max(np.array(colors)) > 1:
            colors = [(c % 256) / 255 * 0.99 + 0.01 for c in colors]
    else:
        colors = get_color_palette(dataset.info['class_count'])
    colors = [np.zeros(3)] + list(map(np.array, colors))  # unknown black

    @lru_cache(maxsize=1000)
    def get_class_representative(label):  # classification
        for x, y in dataset:
            if y == label:
                return x

    def scale01(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def get_frame(datapoint):
        img, lab = datapoint
        classification = np.shape(lab) == ()
        lab_full = np.full(img.shape[:2], lab) if classification else lab
        img_scal = scale01(img)
        black = img_scal * 0
        clab = colorify_label(lab_full + 1, colors)
        label_img = scale01(get_class_representative(datapoint[1])) \
                    if classification else fuse_images(img_scal, clab)
        comp_arr = [[img_scal, black], [label_img, clab]]
        if infer is not None:
            pred = infer(img)
            pred_full = np.full(img.shape[:2], pred) if classification else pred
            cpred = colorify_label(pred_full + 1, colors)
            pred_img = scale01(get_class_representative(pred)) \
                    if classification else fuse_images(img_scal, cpred)
            comp_arr.append([pred_img, cpred])
        comp = compose(comp_arr)

        bar_width, bar_height = comp.shape[1] // 10, comp.shape[0]
        step = bar_height // len(colors)
        bar = np.zeros((bar_height, bar_width), dtype=np.int8)
        for i in range(len(colors)):
            bar[i * step:(i + 1) * step, 1:] = len(colors) - 1 - i
        bar = colorify_label(bar, colors)

        return compose([comp, bar])

    return Viewer().display(dataset, get_frame)


def plot_curves(curves):
    #plt.yticks(np.arange(0, 0.51, 0.05))
    #axes.set_xlim([0, 200])
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.grid(color='0.9', linestyle='-', linewidth=1)
    for name, curve in curves.items():
        plt.plot(curve, label=name, linewidth=1)
    plt.xlabel("broj završenih epoha")
    plt.legend()
    plt.show()
