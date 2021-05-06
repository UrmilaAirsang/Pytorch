import numpy as np
import cv2

class ColorPalette:
    def __init__(self, numColors):
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [128, 0, 255],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [255, 230, 180],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ], dtype=np.uint8)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3), dtype=np.uint8)], axis=0)
            pass

        return

    def getColorMap(self, returnTuples=False):
        if returnTuples:
            return [tuple(color) for color in self.colorMap.tolist()]
        else:
            return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
        pass

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  np.minimum(image[:, :, c] *
                                             (1 - alpha) + alpha * color[c], 255),
                                  image[:, :, c])
    return image
