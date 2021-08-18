import numpy as np
import pandas as pd


def intersections(A: np.ndarray, B: np.ndarray):
    """
    Compute the intersection matrix of two label images

    Args:
    :param A: Input WxH array of labels (of N labels)
    :param B: Input WxH array of labels (of M labels)
    :return: A 2D-array of size NxM
    """
    assert A.shape == B.shape

    # Convert inputs to uint32 so we can add them.
    labels_gt = A.astype(np.uint32)
    labels_pred = B.astype(np.uint32)

    # Count the number of components in each image.
    nlabel_gt = labels_gt.max() + 1
    nlabel_pred = labels_pred.max() + 1
    sz = nlabel_gt * nlabel_pred

    # (Here comes the beautiful part.)
    # Add the images. We have |GT|*|PRED| possible values.
    # Thanks to the multiplication, each pair of components will have its new id.
    H = labels_gt * nlabel_pred + labels_pred
    # Compute the number of pixels for each pair of components.
    # The histogram is sorted by values.
    hist = np.bincount(H.ravel(), minlength=sz)
    # Because the values are sorted in an order we chose, we can reshape the histogram.
    # `hist[i,j]` contains the area of the intersection between components GT_i and PRED_j
    hist = hist.reshape((nlabel_gt, nlabel_pred))
    # hist[0,0] = 1  # <- this is really wrong
    return hist


def compute_bipartite_edge_weigths(intersection_matrix: np.array, mode="jaccard"):
    """[summary]

    Let two sets of components P and Q and a weighted bipartite graph represented as an intersection matrix of size |P|
    × |Q|.
    
    The algorithm selects for each component in P, the best overlap in Q, and vice verse. Only the corresponding edges
    in the graph are kept. 

    Then, each (directed) edge A → B is associated with a weight, either:
 
    * The Jaccard Index (IoU):  w(A,B) = IoU(A,B) = |A ∩ B| / |A ∪ B|
    * The DICE:                 w(A,B) = 2|A ∩ B| / (|A| + |B|)
    * The Coverage:             w(A,B) = Covₐ(B) = |A ∩ B| / |A|  
    * The Target Coverage:      w(A,B) = Cov_b(A) = |A ∩ B| / |B|  

    Args:
        intersection_matrix (np.array): Input |P| × |Q| array with intersection counts
        mode (str, optional): Edge weighting mode. Either ["jaccard" or "coverage", "target_coverage"]. Defaults to "jaccard".

    Returns:
        np.array: A pair (X, Y) with scores for the best match
                  X: P → Q weights (array of size |P|, with IoU(A,B) or Covₐ(B))
                  Y: Q → P weights (array of size |Q|, with IoU(B,A) or Cov_b(A))
    """

    if mode not in {"jaccard", "coverage", "target_coverage", "dice"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    H = intersection_matrix
    nlabel_P, nlabel_Q = H.shape
    # Areas of components
    areas_P = H.sum(axis=1)
    areas_Q = H.sum(axis=0)

    # Select the best overlap
    best_match_P = H.argmax(axis=1)
    best_match_Q = H.argmax(axis=0)

    area_P_Inter_BestMatch = H[np.arange(nlabel_P), best_match_P]
    area_Q_Inter_BestMatch = H[best_match_Q, np.arange(nlabel_Q)]

    if mode == "jaccard":    # normalizer = |A ∪ B| = |A| + |B| - |A ∩ B|
        area_P_normalizer = areas_P + areas_Q[best_match_P] - area_P_Inter_BestMatch
        area_Q_normalizer = areas_Q + areas_P[best_match_Q] - area_Q_Inter_BestMatch
    elif mode == "dice":    # normalizer = 0.5 * (|A| + |B|)
        area_P_normalizer = 0.5 * (areas_P + areas_Q[best_match_P])
        area_Q_normalizer = 0.5 * (areas_Q + areas_P[best_match_Q])
    elif mode == "coverage": # normalizer = |A|
        area_P_normalizer = areas_P
        area_Q_normalizer = areas_Q
    elif mode == "target_coverage": # normalizer = |B|
        area_P_normalizer = areas_Q[best_match_P]
        area_Q_normalizer = areas_P[best_match_Q]

    weights_P = np.devide(area_P_Inter_BestMatch, area_P_normalizer, where = area_P_normalizer > 0)
    weights_Q = np.devide(area_Q_Inter_BestMatch, area_Q_normalizer, where = area_Q_normalizer > 0)

    return weights_P, weights_Q


# def viz_iou(A: np.ndarray, iou: np.ndarray, output_path: str = None, lower_bound=0.5):
#     '''
#     Visualization of the IoU (outputs in a file if ``output`` is not NULL)
#     :param A: Input W*H array of labels (of M labels)
#     :param iou: Input array of M iou values
#     :param lower_bound: 0.5 <= lower_bound < 1
#     '''
#     if not 0.5 <= lower_bound < 1:
#         raise ValueError("0.5 <= lower_bound < 1")
#     cmap = matplotlib.cm.get_cmap(name="RdYlGn")

#     iou_tr = iou.copy()
#     left = iou_tr[iou<=lower_bound]
#     left *= 0.5/lower_bound
#     iou_tr[iou<=lower_bound] = left
#     right = iou_tr[iou>lower_bound]
#     right = (right - lower_bound)/(1-lower_bound) * 0.5 + 0.5
#     iou_tr[iou>lower_bound] = right

#     lut = cmap(iou_tr, bytes=True)[...,:3]
#     lut[0] = (0,0,0)
#     out = lut[A]
#     if output_path:
#         #logging.info("Saving image in %s", output_path)
#         skio.imsave(output_path, out)
#         #logging.info("end saving")
#     else:
#         plt.imshow(out)


def compute_matching_scores(ref: np.ndarray, containder_score: np.ndarray, pairing_threshold = 0.5):
    """
    Compute the F-score, Precision and Recall Scores from IoU scores measured of 2 images components

    The match between two components A and B is considered when the IoU > 0.5. It returns a dataframe with:

            Precision    Recall   F-score
    IoU
    0.500665   0.757566  0.434935  0.552607
    0.502368   0.757244  0.434750  0.552372
    0.505121   0.756600  0.434381  0.551902
    0.506959   0.756278  0.434196  0.551667
    ...
    0.995850   0.000966  0.000555  0.000705
    0.995986   0.000644  0.000370  0.000470
    0.996028   0.000322  0.000185  0.000235
    """
    if ref.size == 0:
        raise ValueError("'ref' parameter is an empty array.")
    if containder_score.size == 0:
        raise ValueError("'containder_score' parameter is an empty array.")

    scores_A = np.sort(ref)
    scores_B = np.sort(containder_score)
    startA = np.searchsorted(scores_A, pairing_threshold, side="right")
    startB = np.searchsorted(scores_B, pairing_threshold, side="right")
    # nonMatchA = startA  # Number of A points with no-match
    # nonMatchB = startB  # Number of B points with no-match
    scores_A1 = scores_A[startA:]
    scores_B1 = scores_B[startB:]
    # We must have a partial bijection between A and B for IoU > 0.5
    assert np.array_equal(scores_A1, scores_B1)

    # P: Size of the prediction set
    # T: Size of the target (reference) set
    # tp: Number of true positive

    P = float(containder_score.size)
    T = float(ref.size)

    iou_values, count = np.unique(scores_A1, return_counts=True)
    tp = np.flipud(np.cumsum(np.flip(count)))

    # Recall = Number of matchs / size(ref)
    # Precision = Number of matchs / size(containder)
    recall = tp / T
    precision = tp / P
    fscore = 2 * tp / (P + T)

    df = pd.DataFrame(
        {
            "IoU": iou_values,
            "Precision": precision,
            "Recall": recall,
            "F-score": fscore,
        }
    )
    return df



