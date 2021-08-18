import numpy as np
from matplotlib import cm, colors
from copy import copy

cmap = copy(cm.get_cmap(name="RdYlGn"))

def colorize_regions(label_map: np.ndarray, label_scores: np.ndarray, lower_bound=0.5):
    '''
    Visualization of the IoU (outputs in a file if ``output`` is not NULL)
    :param label_map: Input W*H array of labels (of M labels)
    :param label_scores: Input array of M iou values
    :param lower_bound: Must be in [0,1]. The center of the colormap (threshold of acceptance)
    '''
    if not (0 < lower_bound < 1):
        raise ValueError("0 < lower_bound < 1")


    ## Create a gamma correction so that lower_bound --> 0.5
    gamma = np.log(0.5) / np.log(lower_bound)
    cmap.set_gamma(gamma)
    lut = cmap(label_scores, bytes=True)[...,:3]
    lut[0] = (0,0,0)
    out = lut[label_map]
    return out




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    lblmap = np.arange(20)
    X = np.linspace(0,1,num=20)
    Y = colorize_regions(lblmap, X, 0.8)
    plt.imshow(Y[np.newaxis, :, :])


