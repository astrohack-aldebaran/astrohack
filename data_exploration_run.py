#imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob
import os.path
from numpy import genfromtxt
import skimage.transform as transform
from scipy import ndimage

datadir = "Sample_Data/SAMPLE/"
outputdir = "Sample_Data/OUTPUT/"

#Functions
def imhist(im):
    # calculates normalized histogram of an image
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h) / (m * n)

def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i + 1]) for i in range(len(h))]

def histeq(im):
    # calculate Histogram
    h = imhist(im)
    cdf = np.array(cumsum(h))  # cumulative distribution function
    sk = np.uint8(255 * cdf)  # finding transfer function values
    s1, s2 = im.shape
    Y = np.zeros_like(im)
    # applying transfered values for each pixels
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    H = imhist(Y)
    # return transformed image, original and new istogram,
    # and transform function
    return Y, h, H, sk

def detectobj(img, blur_radius, threshold):
    imgf = ndimage.gaussian_filter(img, blur_radius)
    labeled, nr_objects = ndimage.label(imgf > threshold)
    if nr_objects == 0:
        return None, [], []
    center = img.shape[0] // 2
    cid = labeled[center, center]
    if len(np.where(labeled == cid)[0]) >= labeled.shape[0] ** 2 / 2:
        # the center of galaxy cannot belong to an object which spans more than 50% of pixels.
        return None, [], []

    tmp = labeled == cid
    x = np.where(tmp.any(axis=0))[0]
    y = np.where(tmp.any(axis=1))[0]
    bound = np.abs(np.array([x[0], x[-1], y[0], y[-1]]) - center).max()
    lower = center - bound
    upper = center + bound
    xs = [lower, lower, upper, upper, lower]
    ys = [lower, upper, upper, lower, lower]
    return labeled, xs, ys


ssdsids = list(set([os.path.split(s)[1].split('-')[0] for s in glob.glob(datadir+"*.csv")]))

#Finding out bounding box for identifying the best image size for deep learning
sizes = []
cropped = []
failed = 0
for idx in ssdsids[0:1000]:
    g_img = genfromtxt(datadir + idx + '-g.csv', delimiter=',')
    log_g_img = np.log10(g_img - g_img.min() + 1e-16)
    imgsize = g_img.shape[0]
    g_img2 = np.uint8(log_g_img)
    y, _, _, _ = histeq(g_img2)
    objs, xs, ys = detectobj(y, 10, (y.max() - y.min())/2)
    if len(xs) != 0 and ys[1] - ys[0] < imgsize:
        sizes.append(ys[1] - ys[0])
        height = sizes[-1]
        c = (g_img[xs[0]:xs[0]+height,xs[0]:xs[0]+height])
        cropped.append(c)
        image = transform.resize(c, (64, 64), 1, 'reflect')
        filename = outputdir + idx + '-' + str(height) + '.png'
        plt.imsave(filename, image)
#         implot = plt.imshow(objs)
#         plt.plot(xs, ys, '-')
#         plt.title(idx)
#         plt.show()
    else:
        failed += 1

#for c in cropped:
    #plt.imshow(transform.resize(c, (64, 64)))
   # plt.show()
