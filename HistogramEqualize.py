import cv2
import numpy as np
import PIL

class HistogramEqualize(object):
    """ Perform histogram equalization on image."""

    def __call__(self, sample):
        return PIL.Image.fromarray(cv2.equalizeHist(np.asarray(sample)))