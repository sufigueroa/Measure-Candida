from skimage.measure import regionprops, find_contours, label
import numpy as np
import pickle
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

def esqueletonize_3D(distance):
    esqueleton = np.zeros(distance.shape)
    for i in range(len(distance)):
        ys, xs = np.where(distance[i] > 0)
        for x, y in zip(xs, ys):
            coor = (i, y, x)
            near_neighbours = get_neigh(coor, distance.shape)
            all_neigh = set(near_neighbours)
            for neigh in near_neighbours:
                all_neigh.update(get_neigh(neigh, distance.shape))
            pixels = [distance[neigh] for neigh in all_neigh]
            # quantile = np.quantile(pixels, .9)
            max_value = np.max(pixels)
            if distance[coor] >= max_value:
                esqueleton[coor] = 255
    return [np.logical_or.reduce(esqueleton)]

def get_contours(hifas, label_hifas, prefix, save_image, equalize, save=False, load=[True, True, True, False]):
    for label_hifa in label_hifas:
        if load[1]:
            with open(f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_hifa.pkl', 'rb') as f:
                hifa = pickle.load(f)
            with open(f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_contour.pkl', 'rb') as f:
                contour = pickle.load(f)
            with open(f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_contour_pos.pkl', 'rb') as f:
                contours_pos = pickle.load(f)
            print(f"Contour ID{label_hifa} loaded!!")
        else:      
            hifa = np.zeros(hifas.shape)
            hifa[hifas == label_hifa] = 1
            contour, contours_pos = only_contours(hifa)
        
            if save:
                with open(f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_contour.pkl', 'wb') as f:
                    pickle.dump(contour, f)
                with open(f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_contour_pos.pkl', 'wb') as f:
                    pickle.dump(contours_pos, f)
                with open(f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_hifa.pkl', 'wb') as f:
                    pickle.dump(hifa, f)
                for i in range(len(contour)):
                    save_image(equalize(contour[i]), f'{PATH}contour_outputs/{prefix}_ID{label_hifa}_{i}_contour.png')
        
        if load[2]:
            with open(f'{PATH}distance_outputs/{prefix}_ID{label_hifa}_distance.pkl', 'rb') as f:
                distance = pickle.load(f)
            print(f"Distance ID{label_hifa} loaded!!")
        else:
            distance = max_dist_3D(hifa, contour, contours_pos)
            if save:
                with open(f'{PATH}distance_outputs/{prefix}_ID{label_hifa}_distance.pkl', 'wb') as f:
                    pickle.dump(distance, f)
                for i in range(len(distance)):
                    save_image(equalize(distance[i]), f'{PATH}distance_outputs/{prefix}_ID{label_hifa}_{i}_distance.png')

        skeleton = esqueletonize_3D(distance)
        if save:
            for i in range(len(skeleton)):
                save_image(equalize(skeleton[i]), f'{PATH}skeleton_outputs/{prefix}_ID{label_hifa}_{i}_skeleton.png')
