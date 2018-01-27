from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value

        colors = OrderedDict({
            "negative": (0,134,0),
            "positive": (18,134,0)})


        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

    def label(self, image, c):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        cv2.imshow("kotrura0",image)

        mask = np.zeros(image.shape[:2], dtype="uint8")

        cv2.drawContours(mask, [c], -1, 255, -1)

        cv2.imshow('kontura', mask)
        masked_data = cv2.bitwise_and(image, image, mask=mask)

        masked_data = cv2.cvtColor(masked_data, cv2.COLOR_BGR2YCrCb)

        plava = cv2.inRange(masked_data, (0, 128, 0), (255, 255, 255))
        plava = cv2.bitwise_not(plava)
        cv2.imshow("plava1", plava)

        nzop = cv2.countNonZero(plava)

        crvena = cv2.inRange(masked_data, (18, 128, 0), (255, 255, 255))
        cv2.imshow("crvena1", crvena)

        nzoc=cv2.countNonZero(crvena)

        if nzop>nzoc:
            return 'negative'
        else:
            return 'positive'
