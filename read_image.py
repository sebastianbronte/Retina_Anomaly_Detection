import numpy as np
import cv2
from scipy.signal import argrelextrema
from scipy import signal
import matplotlib.pyplot as plot


def removeWhiteAlignment(img):
    m = 150
    white = True
    img2 = img.copy()
    upv = img2.shape[0] - m
    lowv = m
    while white:
        idxs = np.argwhere(img2 == 255)
        if len(idxs) == 0:
            white = False
            break
        i = 0
        ok = False
        while i < len(idxs):
            seedpt = (idxs[i][1], idxs[i][0])
            #print ('len(idxs) ' + str(len(idxs)) + ' i ' + str(i) + ' seedpt ' + repr(seedpt) + ' / ' + repr(img2.shape) + ' conditions ' + repr((seedpt[1] < lowv, seedpt[1] > upv)))
            if seedpt[1] < lowv or seedpt[1] > upv:
                cv2.floodFill(img2, None, seedpt, (0,0,0), 3, 3)
                i = len(idxs)
                ok = True
            else:
                i = i + 1
        if not ok and i == len(idxs):
            white = False
    return img2


def computeMask(imcopy2):
    w = imcopy2.shape[1]
    h = imcopy2.shape[0]
    seedptup = ((0,0),(int(w/2),0),(w-1,0),(int(w/2),20))
    inc = 3
    for seedpt in seedptup:
        imcopyback = imcopy2.copy()
        #print('1 mask floodFill ' + repr(seedpt) + ' 255 sum ' + str(np.sum(imcopy2 == 255)) + ' / ' + str(w*h))
        cv2.floodFill(imcopy2, None, seedpt, (255,255,255), inc, inc)
        #print('1 255 after floodfill ' + str(np.sum(imcopy2 == 255)) + ' / ' + str(w*h))
        if (np.sum(imcopy2 == 255) > 0.5*w*h):
            imcopy2 = imcopyback
            inc = 1
            continue
    
    # select the points in a more intelligent way: in case they are brighter than
    # expected, look in the neighbourhood for darker ones.
    # Otherwise the Choroid can be lost
    seedptdown = ((0, h-1), (int(w/2),h-1), (w-1, h-1), (int(w/2), h-20))
    for seedpt in seedptdown:
        imcopyback = imcopy2.copy()
        #print('2 mask floodFill ' + repr(seedpt) + ' 255 sum ' + str(np.sum(imcopy2 == 255)) + ' / ' + str(w*h))
        cv2.floodFill(imcopy2, None, seedpt, (255,255,255), 1, 1)
        #print('2 255 after floodfill ' + str(np.sum(imcopy2 == 255)) + ' / ' + str(w*h))
        if (np.sum(imcopy2 == 255) > 0.9*w*h):
            imcopy2 = imcopyback
            continue
    
    imcopy2 = cv2.dilate(imcopy2, None, iterations = 3)
    imcopy2 = cv2.erode(imcopy2, None,iterations = 3)
    th = 254
    _, imcopy2 = cv2.threshold(imcopy2, th, 255, cv2.THRESH_BINARY_INV)
    return imcopy2


def detectminmax(img, mask, imshow, imwrite):
    minmaximg = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
    mina = []
    maxa = []
    fs = 1000.0  # Sampling frequency
    fc = 60.0 #30  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    c, b = signal.butter(5, w, 'low')
    for n in range(masked_original.shape[1]):

        a = masked_original[:,n]
        a2 = signal.filtfilt(c, b, a)

        mina2 = argrelextrema(a2, np.less)[0]
        maxa2 = argrelextrema(a2, np.greater)[0]
        
        minav2 = mina2.copy()
        maxav2 = maxa2.copy()
        for i in range(len(mina2)):
            minav2[i] = a2[mina2[i]]
            pt = (n, mina2[i])
            if (mask[pt[1]][pt[0]] == 0):
                continue
            mina.append(pt)
            cv2.circle(minmaximg, pt, 1,(255,255,255),1)
        for i in range(len(maxa2)):
            maxav2[i] = a2[maxa2[i]]
            pt = (n, maxa2[i])
            if (mask[pt[1]][pt[0]] == 0):
                continue
            maxa.append(pt)
            cv2.circle(minmaximg, pt, 1,(128,128,128),1)
        
        if imshow and n == 20:
            fig, ax = plot.subplots()
            ax.plot(a)
            ax.plot(a2)
            ax.scatter(mina2, minav2)
            ax.scatter(maxa2, maxav2)
            ax.grid
            plot.show()
            if imwrite:
                plot.savefig('column_evaluation_20_' + str(ex) + '.png')

    return minmaximg, mina, maxa

def estimateLayerTraces(traces):
    #print(repr(traces))
    #transform from list of tuples to np.array, now we know the total size
    margin = 2
    lent = len(traces)
    trarray = np.zeros((lent, 2))
    traces = []
    for i in range(lent):
        trarray[i][0] = traces[i][0]
        trarray[i][1] = traces[i][1]
        
    for i in range(lent):
        pt1 = trarray[i]
        for j in range(lent):
            if j == i: continue
            idtmp = trarray[:,0] == pt[0] + 1
            idtmp2 = trarray[idtmp,1] > pt[1] - margin and trarray[idtmp,1] < pt[1] + margin
            pt2 = trarray[np.floor(len(trarray[idtmp2])/2)]

    print(repr(trarray))

def computeStats(img):
    average = np.mean(img)
    median = np.median(img)
    stdev = np.std(img)
    return average, median, stdev


def imgExamples():
    testIdx = {}
    testIdx[1] = 'samples/normal/Image 5.TIFF'
    testIdx[2] = 'samples/not_normal/DME-Image 11.TIFF'
    testIdx[3] = 'samples/not_normal/CNV-624911-3.jpeg'
    testIdx[4] = 'samples/not_normal/CNV-53018-10.jpeg'
    return testIdx


def computeHistogram(imcopy, nbins, imshow, imwrite):
    # compute the histogram (for later, if necessary)
    imcopyrgb = cv2.cvtColor(imcopy, cv2.COLOR_GRAY2RGB)
    f = 5
    h = np.zeros((300,(nbins+1)*f,3))
    if len(imcopy.shape) != 2:
        imcopy = cv2.cvtColor(imcopy, cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([imcopy],[0], None, [nbins], [0,256])
    cv2.normalize(hist_item, hist_item, 0,255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    #print(repr(hist))

    for x,y in enumerate(hist):
        cv2.line(h, (f*x,0), (f*x,y), (255,255,255), f)
    y = np.flipud(h)
    if imshow:
        cv2.imshow('histogram', y)
    if imwrite:
        cv2.imwrite('histogram_' + str(ex) + '.jpg', y)
    
    return hist

def computeAdaptiveThresholdFromHistogram(hist, nbins, val_in_es_coef = 0.001):
    s = np.sum(hist)
    val = 0
    for x in range(len(hist)):
        val += hist[-x-1]
        #print (str(x-1) + ' ' + str(len(hist)-x-1) + ' ' + str(s) + ' ' + str(val) + ' ' + str(val/s))
        if val/s > val_in_es_coef:
            th = 256/nbins*(len(hist)-x-1)
            break
    #print(str(th))
    return th


def computeAlignmentMask(imcopy, th, imshow, imwrite):
    _, thresh_img = cv2.threshold(imcopy, th, 255, cv2.THRESH_BINARY)
    if imshow:
        cv2.imshow('thresholded', thresh_img)

    #look for the seed point
    seedPoints = set()
    for u in range(thresh_img.shape[0]):
        for v in range(thresh_img.shape[1]):
            pix = thresh_img[u][v]
            if pix != 0:
                seedPoints.add((v, u))



    #compute the sobel image
    sob = cv2.Sobel(imcopy, -1, 0,1,5)
    if imshow:
        cv2.imshow('Sobel', sob)
    if imwrite:
        cv2.imwrite('Sobel_' + str(ex) + '.jpg', sob)


    #infer the mask to which the algorithm will be applied
    imcopy2 = median.copy()
    imcopy2 = computeMask(imcopy2)
    if imshow:
        cv2.imshow('maskTrial',imcopy2)
    if imwrite:
        cv2.imwrite('maskTrial_' + str(ex) + '.jpg', imcopy2)

    # TODO: review mask computation, for the first image cuts, ok, not for the others
    return imcopy2



if __name__ == '__main__':
    testIdx = imgExamples()

    #read images (ex = test number)
    ex = 1
    imwrite=False
    imshow=False
    im = cv2.imread(testIdx[ex], 0) #grayscale
    #print('im size ' + repr(im.shape))

    if imshow:
        cv2.imshow('original',im)
    if imwrite:
        cv2.imwrite('original_' + str(ex) + '.jpg', im)


    # remove the artifacts due to image alignment
    im2 = removeWhiteAlignment(im)

    if imshow:
        cv2.imshow('original rectified', im2)
    if imwrite:
        cv2.imwrite('original_rectified_' + str(ex) + '.jpg', im2)

    # test to check the amount of noise in the image (to see if the median filter would be enough or not)
    # compute the median, average, std and other high magnitude stats on the rectified image
    mean, median, std = computeStats(im2)
    print('mean ' + repr(mean) + ' median ' + repr(median) + ' std ' + repr(std))


    # apply the median filter to the image, to reduce the salt and pepper noise
    median = cv2.medianBlur(im2, 5)
    if imshow:
        cv2.imshow('median', median)
    if imwrite:
        cv2.imwrite('median_' + str(ex) + '.jpg', median)

    imcopy = median.copy()
    nbins = 256
    hist = computeHistogram(median, nbins, imshow, imwrite)

    th = computeAdaptiveThresholdFromHistogram(hist, nbins)

    #improvements. Some of the images can be aligned with some white on the image. first detect the white and fill in with black, then create the mask
    imcopy2 = computeAlignmentMask(imcopy, th, imshow, imwrite)

    #apply mask to the corrected median image
    masked_median = cv2.bitwise_and(median, imcopy2)# median, mask=imcopy2)

    if imshow:
        cv2.imshow('masked median', masked_median)
    if imwrite:
        cv2.imwrite('masked_median_' + str(ex) + '.jpg', imcopy2)

    # apply sobel to have an idea of where the layer delimiters go.
    sob2 = cv2.Sobel(masked_median, -1, 0,1,5)
    #print('sobel min ' + repr(np.min(sob2)) + ' max ' + repr(np.max(sob2)))
    _, thsob2 = cv2.threshold(sob2,15, 255, cv2.THRESH_BINARY)
    if imshow:
        cv2.imshow('Sobel2', thsob2)
    if imwrite:
        cv2.imwrite('Sobel2_' + str(ex) + '.jpg', thsob2)

    masked_original = cv2.bitwise_and(im2, imcopy2)
    if imshow:
        cv2.imshow('masked original',masked_original)
    if imwrite:
        cv2.imwrite('masked_original_' + str(ex) + '.jpg', masked_original)


    #compute maxs and mins from the image colums
    minmaximg_original, mina_orig, maxa_orig = detectminmax(masked_original, imcopy2, imshow, imwrite)
    minmaximg_median, mina_median, maxa_median = detectminmax(masked_median, imcopy2, imshow, imwrite)

    # represent the minimum and maximum points on the corresponding images
    if imshow:
        cv2.imshow('minmax img ori', minmaximg_original)
        cv2.imshow('minmax img med', minmaximg_median)

    if imwrite:
        cv2.imwrite('minmax_img_ori_' + str(ex) + '.jpg', minmaximg_original)
        cv2.imwrite('minmax_img_med_' + str(ex) + '.jpg', minmaximg_median)

    # estimate Layer Traces out of detected minimum and maximum points. Not finished.
    #traces_min_orig = estimateLayerTraces(mina_orig)
    #traces_max_orig = estimateLayerTraces(maxa_orig)
    #traces_min_median = estimateLayerTraces(mina_median)
    #traces_max_median = estimateLayerTraces(maxa_median)

    #traces_orig = traces_min_orig.copy()
    #traces_orig.append(traces_max_orig)
    #traces_median = traces_min_median.copy()
    #traces_median.append(traces_max_median)

    #print('mina_orig' + repr(mina_orig))
    #print('maxa_orig' + repr(maxa_orig))
    #print('mina_median' + repr(mina_median))
    #print('maxa_median' + repr(maxa_median))

    # TODO: from here explore contour detection within images, possible corrections as there can be cuts in the path of the contours, etc.

    # Then, approximate the detected functions and model normal layer curves with a polinomial.

    # Finally, compare the normal ones with the detected polinomial functions and check how close they are.

    cv2.waitKey(0)
    cv2.destroyAllWindows()
