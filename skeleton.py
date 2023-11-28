from skimage.measure import regionprops, find_contours, label
import numpy as np

PATH = 'results/'

def only_contours(hifa):
    new_hifa = np.zeros(hifa.shape)
    for i in range(len(hifa)):
        labels = label(hifa[i])
        for region in regionprops(labels):
            contour = find_contours(labels == region.label, 0.5)[0]
            for point in contour:
               new_hifa[i, int(point[1]), int(point[0])] = 1
    return new_hifa.astype(np.uint8)

def max_dist_3D(hifa, contour):
    Z, Y, X = hifa.shape
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                pass


    return contour

def esqueletonize_3D(hifa, contour):
    return max_dist_3D(hifa, contour)


def get_contours(hifas, label_hifas, prefix, save_image, equalize):
    for label_hifa in label_hifas:
        hifa = np.zeros(hifas.shape)
        hifa[hifas == label_hifa] = 1
        contour = only_contours(hifa)
        # for i in range(len(contour)):
        #     save_image(equalize(contour[i]), f'{PATH}{prefix}_ID{label_hifa}_{i}_contour.png')
        esqueletonize_3D(hifa, contour)

