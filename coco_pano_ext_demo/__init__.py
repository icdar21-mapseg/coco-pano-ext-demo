__version__ = "1.0.0"


from .coco import COCO, COCO_plot, precision_recall_maps
from .prec_recall_map import cmap as _cmap

__all__ = ["COCO", "COCO_plot", "precision_recall_maps"]
