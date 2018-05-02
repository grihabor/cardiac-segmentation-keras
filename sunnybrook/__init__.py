import os

from .training import train
from .contour import load_all_contours, export_all_contours
from helpers import get_SAX_SERIES

def load_data(dataset_path, contour_type, crop_size):
    contours_path = os.path.join(dataset_path, 'contours')
    images_path = os.path.join(dataset_path, 'images')
    
    contours = load_all_contours(contours_path, contour_type, shuffle=True)
    return export_all_contours(
        contours,
        images_path,
        crop_size=crop_size,
        sax_series=get_SAX_SERIES(),
    )

