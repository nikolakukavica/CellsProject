import numpy as np
import cv2
from test import ColorLabeler
import os

dirname = 'images'
pictures = [dirname+'/'+name for name in os.listdir('./'+dirname)]

print pictures

#pictures = ['1 (11).jpg', '1 (16).jpg', '1 (17).jpg', '1 (18).jpg', '1 (23).jpg', '1 (30).jpg', '1 (68).jpg', '1 (71).jpg']
#pictures = ['images/1 (19).jpg', 'images/1 (30).jpg', 'images/1 (15).jpg', 'images/1 (2).jpg']
number = 0
show_pictures=False

broj_slike = [1,2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,28,29,30,31,33,34,37,38,39,40,41,43,44,45,46,47,48,52,53,54,55,56,57,58,59,60,61,62,63,65,68,69,71,72,73,75,76,77,79,85,86,87,88,89,90,91,92]
broj_jedara = [40,28,145,112,36,13,91,1516,362,419,257,228,121,136,110,856,819,964,885,928,842,915,770,164,150,43,23,44,52,52,147,174,92,63,136,97,113,112,44,42,50,41,35,159,131,178,95,102,100,100,53,100,70,96,95,100,19,27,367,6,38,26,50,71,34,35,26,37]

print len(broj_slike) - len(broj_jedara)

def nothing(x): #needed for createTrackbar to work in python.
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('temp', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Fail', 'temp', 0, 255, nothing)
cv2.createTrackbar('Blur', 'temp', 5, 10, nothing)
cv2.createTrackbar('Kernel', 'temp', 3, 20, nothing)
#bilo 12
cv2.createTrackbar('Clip Limit', 'temp', 14, 20, nothing)
cv2.createTrackbar('Title Grid Size', 'temp', 40, 40, nothing)
#2,50
cv2.createTrackbar('Alpha', 'temp', 2, 20, nothing)
cv2.createTrackbar('Beta', 'temp', 50, 100, nothing)

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

    brojac = 0
    for num, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        #mask = np.zeros(image.shape[:2], dtype="uint8")

        #cv2.drawContours(temp_color, [cnt], -1, 255, -1)


        if radius > 5:
            brojac += 1
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
                cv2.drawContours(temp_color, [cnt], -1, (255, 255, 0), 0)
            else:
                cv2.drawContours(temp_color, [cnt], -1, (0, 255, 255), 0)

            cv2.putText(temp_color, str(text), (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.imshow("Labeled image", temp_color)

    #print ("Number of detected objects on picture is: ",brojac)
    return brojac


def test(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    cv2.imshow('l_channel', l)
    cv2.imshow('a_channel', a)
    cv2.imshow('b_channel', b)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    cv2.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    cv2.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)

    konj = np.add(v,l)
    cv2.imshow('konj',konj)

    return final

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

    #lab = cv2.cvtColor(temporary.copy(), cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)

    #cv2.imshow("L_channel0", l)
    #cv2.imshow("A_channel0", a)
    #cv2.imshow("B_channel0",b)

    #alpha = 1.5
    alpha = cv2.getTrackbarPos('Alpha', 'temp')
    beta = cv2.getTrackbarPos('Beta', 'temp')

    h, s, v = cv2.split(hsv)

    #cv2.imshow("H_channel0", h)
    #cv2.imshow("S_channel0", s)
    #cv2.imshow("V_channel0",v)

    hsv = cv2.convertScaleAbs(hsv,alpha=1,beta=0)

    h, s, v = cv2.split(hsv)

    #cv2.imshow("H_channel", h)
    #cv2.imshow("S_channel", s)
    cv2.imshow("V_channel",v)

    #bgr = cv2.imshow(hsv,cv2.COLOR_HSV2BGR)
    #g = cv2.imshow(bgr, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("siva",g)

    grayscale = cv2.cvtColor(temporary, cv2.COLOR_BGR2GRAY)

    grayscale = cv2.convertScaleAbs(grayscale, alpha=alpha, beta=beta)



    #global histrogram
    #img = cv2.equalizeHist(grayscale)

    img = grayscale.copy()

    img0 = v.copy()

    #img = v.copy()

    #local histogram (devide image into regions and then do histogram equlization) 2.0 i 8,8
    cl = cv2.getTrackbarPos('Clip Limit', 'temp')
    tgs = cv2.getTrackbarPos('Title Grid Size', 'temp')
    clahe = cv2.createCLAHE(clipLimit=float(cl), tileGridSize=(tgs, tgs))
    img = clahe.apply(img)



    #img = v.copy()

    (mu, sigma) = cv2.meanStdDev(img)
    (mu0, sigma0) = cv2.meanStdDev(img)

    blur = cv2.getTrackbarPos('Blur', 'temp')
    if blur % 2 !=0:
        blurred = cv2.GaussianBlur(img, (blur, blur), sigmaX=sigma, dst=mu)
        blurred0 = cv2.GaussianBlur(img0, (blur, blur), sigmaX=sigma0, dst=mu0)
    else:
        blurred = cv2.GaussianBlur(img, (blur+1, blur+1), sigmaX=sigma, dst=mu)
        blurred0 = cv2.GaussianBlur(img0, (blur, blur), sigmaX=sigma0, dst=mu0)

    # compute the median of the single channel pixel intensities
    v = np.median(blurred)
    sigma=0.33
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    cv2.imshow("edges",edged)

    #2, 11
    th1 = cv2.adaptiveThreshold(blurred0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -1)

    #th2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

    th3 = cv2.adaptiveThreshold(blurred0, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -1)

    #th4 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)

    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = cv2.getTrackbarPos('Kernel', 'temp')
    if kernel_size != 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=2)
        th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=2)

        #th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=2)
        #th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=2)

        th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel, iterations=2)
        th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=2)

        #th4 = cv2.morphologyEx(th4, cv2.MORPH_OPEN, kernel, iterations=2)
        #th4 = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel, iterations=2)

        #labeling_objects(temporary.copy(), thresh.copy())

        #watershed(thresh.copy(),temporary.copy())

        th=cv2.bitwise_or(thresh,cv2.bitwise_or(th1,th3))

        cv2.imshow("end",th)

        #labeling_objects(temporary.copy(), th.copy())



        #redni_broj = picture.split("(")
        #redni_broj = redni_broj[1].split(")")
        #indeks = broj_slike.index(int(redni_broj[0]))

        # print (redni_broj,indeks)

        #cell_number1 = labeling_objects(temporary.copy(), th.copy())
        #cell_number2 = labeling_objects(temporary.copy(), thresh.copy())
        #print ("Slika pod nazivom:", picture, "Realan broj celija:", str(broj_jedara[indeks]), " dobijeno adaptivnim tresholdom: ",
               #str(cell_number1)," procenat: ",cell_number1/float(broj_jedara[indeks]), " dboijeno orsuom: ", str(cell_number2)," procenat: ",cell_number2/float(broj_jedara[indeks]))


    cv2.imshow("th1",th1)
    #cv2.imshow("th2", th2)
    cv2.imshow("th3", th3)
    cv2.imshow("th",cv2.bitwise_or(th1,th3))
    cv2.imshow("thwithcanny", cv2.bitwise_or(cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2),cv2.bitwise_or(th1,th3)))
    cv2.imshow("thresh", thresh)

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