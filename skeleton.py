from skimage.measure import regionprops, find_contours, label
import numpy as np
from collections import deque

PATH = 'results/'

def only_contours(hifa):
    new_hifa = np.zeros(hifa.shape)
    contours = []
    for i in range(len(hifa)):
        labels = label(hifa[i])
        for region in regionprops(labels):
            for contour in find_contours(labels == region.label, 0.5):
                contour = [(i, int(c[0]), int(c[1])) for c in contour]
                contours.extend(contour)
                for point in contour:
                    new_hifa[i, point[1], point[2]] = 1
    return new_hifa.astype(np.uint8), contours

def get_neigh(coor, size):
    neigh = []
    if 0 < coor[0]:
        neigh.append((coor[0] - 1, coor[1], coor[2]))
    if 0 < coor[1]:
        neigh.append((coor[0], coor[1] - 1, coor[2]))
    if 0 < coor[2]:
        neigh.append((coor[0], coor[1], coor[2] - 1))
    if coor[0] < size[0] - 1:
        neigh.append((coor[0] + 1, coor[1], coor[2]))
    if coor[1] < size[1] - 1:
        neigh.append((coor[0], coor[1] + 1, coor[2]))
    if coor[2] < size[2] - 1:
        neigh.append((coor[0], coor[1], coor[2] + 1))

    # Flor 3D (abajo)
    if (0 < coor[0]) and (0 < coor[1]):
        neigh.append((coor[0] - 1, coor[1] - 1, coor[2]))
    if (0 < coor[0]) and (0 < coor[2]):
        neigh.append((coor[0] - 1, coor[1], coor[2] - 1))
    if (0 < coor[0]) and (coor[1] < size[1] - 1):
        neigh.append((coor[0] - 1, coor[1] + 1, coor[2]))
    if (0 < coor[0]) and (coor[2] < size[2] - 1):
        neigh.append((coor[0] - 1, coor[1], coor[2] + 1))
    
    # Flor 3D (arriba)
    if (coor[0] < size[0] - 1) and (0 < coor[1]):
        neigh.append((coor[0] + 1, coor[1] - 1, coor[2]))
    if (coor[0] < size[0] - 1) and (0 < coor[2]):
        neigh.append((coor[0] + 1, coor[1], coor[2] - 1))
    if (coor[0] < size[0] - 1) and (coor[1] < size[1] - 1):
        neigh.append((coor[0] + 1, coor[1] + 1, coor[2]))
    if (coor[0] < size[0] - 1) and (coor[2] < size[2] - 1):
        neigh.append((coor[0] + 1, coor[1], coor[2] + 1))

    # Diagonales
    if (0 < coor[1]) and (0 < coor[2]):
        neigh.append((coor[0], coor[1] - 1, coor[2] - 1))
    
    if (0 < coor[1]) and (coor[2] < size[2] - 1):
        neigh.append((coor[0], coor[1] - 1, coor[2] + 1))

    if (0 < coor[2]) and (coor[1] < size[1] - 1):
        neigh.append((coor[0], coor[1] + 1, coor[2] - 1))
    
    if (coor[1] < size[1] - 1) and (coor[2] < size[2] - 1):
        neigh.append((coor[0], coor[1] + 1, coor[2] + 1))

    # Diagonales 3D (abajo)
    if (0 < coor[0]) and (0 < coor[1]) and (0 < coor[2]):
        neigh.append((coor[0] - 1, coor[1] - 1, coor[2] - 1))
    
    if (0 < coor[0]) and (0 < coor[1]) and (coor[2] < size[2] - 1):
        neigh.append((coor[0] - 1, coor[1] - 1, coor[2] + 1))

    if (0 < coor[0]) and (0 < coor[2]) and (coor[1] < size[1] - 1):
        neigh.append((coor[0] - 1, coor[1] + 1, coor[2] - 1))
    
    if (0 < coor[0]) and (coor[1] < size[1] - 1) and (coor[2] < size[2] - 1):
        neigh.append((coor[0] - 1, coor[1] + 1, coor[2] + 1))
    
    # Diagonales 3D (arriba)
    if (coor[0] < size[0] - 1) and (0 < coor[1]) and (0 < coor[2]):
        neigh.append((coor[0] + 1, coor[1] - 1, coor[2] - 1))
    
    if (coor[0] < size[0] - 1) and (0 < coor[1]) and (coor[2] < size[2] - 1):
        neigh.append((coor[0] + 1, coor[1] - 1, coor[2] + 1))

    if (coor[0] < size[0] - 1) and (0 < coor[2]) and (coor[1] < size[1] - 1):
        neigh.append((coor[0] + 1, coor[1] + 1, coor[2] - 1))
    
    if (coor[0] < size[0] - 1) and (coor[1] < size[1] - 1) and (coor[2] < size[2] - 1):
        neigh.append((coor[0] + 1, coor[1] + 1, coor[2] + 1))
    

    return neigh

def max_dist_3D(hifa, contour, contours_pos):
    Z, Y, X = hifa.shape
    start = deque(contours_pos)
    points_to_visit = deque()

    while len(start) > 0:
        point = start.popleft()
        neighbours = get_neigh(point, (Z, Y, X))
        for neigh in neighbours:
            if hifa[neigh] and not contour[neigh]:
                points_to_visit.append(neigh)

    while len(points_to_visit) > 0:
        point = points_to_visit.popleft()
        if contour[point]: continue
        neighbours = get_neigh(point, (Z, Y, X))
        min_dist = None
        for neigh in neighbours:
            if not hifa[neigh]: continue
            if not contour[neigh]:
                points_to_visit.append(neigh)
                continue
            distance = contour[neigh]
            if (min_dist == None) or (distance < min_dist):
                min_dist = distance
        if min_dist == None:
            min_dist = np.max(contour)
        contour[point] = min_dist + 1
    
    return contour

def esqueletonize_3D(hifa, contour, contours_pos):
    return max_dist_3D(hifa, contour, contours_pos)

def get_contours(hifas, label_hifas, prefix, save_image, equalize, save=False):
    for label_hifa in label_hifas:
        hifa = np.zeros(hifas.shape)
        hifa[hifas == label_hifa] = 1

        contour, contours_pos = only_contours(hifa)
        if save:
            for i in range(len(contour)):
                save_image(equalize(contour[i]), f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_{i}_contour.png')
        
        distance = esqueletonize_3D(hifa, contour, contours_pos)
        if True:
            for i in range(len(distance)):
                save_image(equalize(distance[i]), f'{PATH}distance_outputs/{prefix}_ID{label_hifa}_{i}_distance.png')
