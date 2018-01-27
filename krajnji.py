import numpy as np
import cv2
from test import ColorLabeler
import os

dirname = 'images'
pictures = [dirname+'/'+name for name in os.listdir('./'+dirname)]

print pictures

#pictures = ['1 (11).jpg', '1 (16).jpg', '1 (17).jpg', '1 (18).jpg', '1 (23).jpg', '1 (30).jpg', '1 (68).jpg', '1 (71).jpg']
number = 0
show_pictures=True

def nothing(x): #needed for createTrackbar to work in python.
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('temp', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Fail', 'temp', 0, 255, nothing)
cv2.createTrackbar('Blur', 'temp', 5, 10, nothing)
cv2.createTrackbar('Kernel', 'temp', 3, 20, nothing)
cv2.createTrackbar('Clip Limit', 'temp', 12, 20, nothing)
cv2.createTrackbar('Title Grid Size', 'temp', 8, 20, nothing)

cv2.namedWindow('hlp', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Fail', 'hlp', 0, 255, nothing)
#mozda 7, 6 definitivno
cv2.createTrackbar('Peak','hlp',2,10,nothing)
cv2.createTrackbar('Border','hlp',31,50,nothing)
cv2.createTrackbar('Gap','hlp',7,50,nothing)

def watershed(thresh, temporary):
    border = cv2.getTrackbarPos('Border', 'hlp')
    gap = cv2.getTrackbarPos('Gap', 'hlp')
    peak = cv2.getTrackbarPos('Peak', 'hlp')

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_normalized = cv2.normalize(dist, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    distborder = cv2.copyMakeBorder(dist, border, border, border, border,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (border - gap) + 1, 2 * (border - gap) + 1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)

    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx * (1.0 / peak), 255, cv2.THRESH_BINARY)

    peaks8u = cv2.convertScaleAbs(peaks)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pek = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, kernel, iterations=1)

    pek = cv2.morphologyEx(pek, cv2.MORPH_DILATE, kernel, iterations=4)

    peaks8u = cv2.convertScaleAbs(pek)


    sure_bg = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

    sure_fg = peaks8u
    unknown = cv2.subtract(sure_bg, sure_fg)



    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(temporary, markers)
    temporary[markers == -1] = [0, 255, 255]

    if show_pictures:
        cv2.imshow("Distance normalized", dist_normalized)
        cv2.imshow("Peaks", peaks)
        cv2.imshow("Peaks after morphology", pek)
        cv2.imshow("Sure foreground",sure_fg)
        cv2.imshow("Sure background",sure_bg)
        cv2.imshow("Unknown regions",unknown)
    else:
        cv2.destroyWindow("Distance normalized")
        cv2.destroyWindow("Peaks")
        cv2.destroyWindow("Peaks after morphology")
        cv2.destroyWindow("Sure foreground")
        cv2.destroyWindow("Sure background")
        cv2.destroyWindow("Unknown regions")

    cv2.imshow("Result", temporary)

def labeling_objects(temp_color, temp_binary):
    simage, contours, hierarchy = cv2.findContours(temp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ratio = temp_color.shape[0] / float(temp_color.shape[0])

    cl = ColorLabeler()
    """
    priv = cv2.cvtColor(temp_color, cv2.COLOR_BGR2YCrCb)

    cv2.imshow("priv",priv)

    plava = cv2.inRange(priv, (0,128,0), (255,255,255))
    plava = cv2.bitwise_not(plava)

    cv2.imshow("plava",plava)

    crvena = cv2.inRange(priv, (18, 128, 0), (255, 255, 255))
    crvena = cv2.bitwise_not(crvena)

    cv2.imshow("crvena", crvena)
    """

    for num, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)

        color = cl.label(temp_color.copy(), cnt)

        cnt = cnt.astype("float")
        cnt *= ratio
        cnt = cnt.astype("int")
        text = "{}".format(num)
        #text = "{} - {}".format(num,color)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if color=="negative":
            cv2.drawContours(temp_color, [cnt], 0, (255, 255, 0), 0)
        else:
            cv2.drawContours(temp_color, [cnt], 0, (0, 255, 255), 0)

        cv2.putText(temp_color, str(text), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    print ("Number of detected objects on picture is: ",len(contours))
    cv2.imshow("Labeled image",temp_color)

while True:

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        quit()
    elif key == ord('a'):
        print "next left picture"
        number = (number - 1) % len(pictures)
    elif key == ord('d'):
        print "next right picture"
        number = (number + 1) % len(pictures)
    elif key == ord('q'):
        if show_pictures:
            print "hide pictures"
            show_pictures=False
        else:
            print "show pictures"
            show_pictures=True

    picture = pictures[number]

    temporary = cv2.imread(picture, 1)
    temporary = cv2.resize(temporary, (1024, 1024), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(temporary.copy(), cv2.COLOR_BGR2HSV)

    grayscale = cv2.cvtColor(temporary, cv2.COLOR_BGR2GRAY)

    #global histrogram
    #img = cv2.equalizeHist(grayscale)
    img = grayscale.copy()

    #local histogram (devide image into regions and then do histogram equlization) 2.0 i 8,8
    cl = cv2.getTrackbarPos('Clip Limit', 'temp')
    tgs = cv2.getTrackbarPos('Title Grid Size', 'temp')
    clahe = cv2.createCLAHE(clipLimit=float(cl), tileGridSize=(tgs, tgs))
    img = clahe.apply(img)

    (mu, sigma) = cv2.meanStdDev(img)
    blur = cv2.getTrackbarPos('Blur', 'temp')
    if blur % 2 !=0:
        blurred = cv2.GaussianBlur(img, (blur, blur), sigmaX=sigma, dst=mu)
    else:
        blurred = cv2.GaussianBlur(img, (blur+1, blur+1), sigmaX=sigma, dst=mu)

    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = cv2.getTrackbarPos('Kernel', 'temp')
    if kernel_size != 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        labeling_objects(temporary.copy(), thresh.copy())
        #watershed(thresh.copy(),temporary.copy())

    if show_pictures:
        cv2.imshow("Grayscale", grayscale)
        cv2.imshow("Equalized image",img)
        cv2.imshow("Blurred image", blurred)
        cv2.imshow("Otsu thresholded", thresh)
    else:
        cv2.destroyWindow("Grayscale")
        cv2.destroyWindow("Equalized image")
        cv2.destroyWindow("Blurred image")
        cv2.destroyWindow("Otsu thresholded")



cap.release()
cv2.destroyAllWindows()
"""
def pomocna():
    color = ('b', 'g', 'r')
    #for i, col in enumerate(color):
        #hist_full = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        #plt.plot(hist_full, color=col)
        #plt.xlim([0, 256])
    #plt.show()

    hist_full = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    elem = np.argmax(hist_full)
    print (elem)

    kvazimodo, th1 = cv2.threshold(hsv, elem, 255, cv2.THRESH_BINARY)

    cv2.imshow("kvazimodo",th1)
    cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX)
    plt.plot(hist_full)
    plt.show()
"""