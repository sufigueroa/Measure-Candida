import numpy as np
import math
from PIL import Image
import cv2
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.morphology import skeletonize
from collections import deque

import skeleton as ske

PATH = 'results/'
###################################################
### Funciones Generales
###################################################

# https://stackoverflow.com/questions/18602525/python-pil-for-loop-to-work-with-multi-image-tiff
# Funcion para abrir la imagen tiff y dejarla como una matriz
def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def save_image(img, path):
    img = Image.fromarray(np.uint8(img), 'L')
    img.save(path)

def save_rgb_image(img, path):
    img = Image.fromarray(np.uint8(img), 'RGB')
    img.save(path)

def save_video(frames, path):
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frames[0].shape[1], frames[0].shape[0]), 0)
    for frame in frames:
        video.write(np.uint8(frame))
    video.release()

###################################################


###################################################
### Funciones de Interpolacion
###################################################

def isotropic_interpolation(matrix):
    new = []
    for i in range(len(matrix) - 1):
        img1 = matrix[i]
        img2 = matrix[i + 1]
        ab = np.dstack((img1, img2))
        inter = np.mean(ab, axis=2, dtype=ab.dtype) 
        new.append(img1)
        new.append(inter)
    new.append(img2)
    return np.array(new)

###################################################

def equalize(matrix):
    max_value = np.max(matrix)
    return matrix / max_value * 255

def initial_segmentation(img, umbral, prefix):
    if prefix == "med":
        segmented = (img > np.median(img)) * 255
    else:
        segmented = (img > umbral) * 255
    return segmented.astype(np.uint8)

def denoise(img):
    k_first_erosion            = np.ones((3,3), np.uint8)
    k_first_dilation           = np.ones((2,2), np.uint8)
    k_second_dilation          = np.ones((15,15), np.uint8)
    k_second_erosion           = np.ones((9,9), np.uint8)

    img_dilation = cv2.dilate(img, k_first_dilation) 
    img_erosion = cv2.erode(img, k_first_erosion)
    img_dilation = cv2.dilate(img_erosion, k_second_dilation) 
    img_erosion = cv2.erode(img_dilation, k_second_erosion)
    return img_erosion

def get_mask(index, matrix, prefix, OR=True):   
    segmented = initial_segmentation(matrix[index], 40, prefix)
    return denoise(segmented)

def segmentation(index, matrix, prefix, save=False):
    if save: save_image(matrix[index], f'{PATH}normal_{index}.png')
    mask = get_mask(index, matrix, prefix, OR=False)
    if save: save_image(mask, f'{PATH}{prefix}_mask_{index}.png')
    return mask

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
    return neigh

def grow_from_coor(matrix, bitmap, region_id, coor):
    search = deque([coor])
    while 0 < len(search):
        point = search.popleft()
        if matrix[point] and not bitmap[point]:
            bitmap[point] = region_id
            search.extend(get_neigh(point, matrix.shape))
    return bitmap

def region_growing3D(matrix):
    Z, Y, X = matrix.shape
    bitmap = np.zeros(matrix.shape)
    region = 1
    for z in range(Z):
        for j in range(Y):
            for i in range(X):
                if not bitmap[z,j,i] and matrix[z,j,i]:
                    bitmap = grow_from_coor(matrix, bitmap, region, (z, j, i))
                    region += 1
    return bitmap

def get_props_per_region(regions_found):
    props_dict = {i+1: {"area": 0, "eccentricity": [], "perimeter": 0} for i in range(int(np.max(regions_found)))}
    for found_region in regions_found.astype(np.uint8):
        for region in regionprops(found_region):
            region_id = region.label
            props_dict[region_id]["area"] += region.area
            props_dict[region_id]["perimeter"] += region.perimeter
            props_dict[region_id]["eccentricity"].append(region.eccentricity)
    return props_dict

def remove_noise(regions_found, props_dict):
    to_delete = []
    for region in props_dict.keys():
        if props_dict[region]["area"] < 100:
            regions_found[regions_found == region] = 0
            to_delete.append(region)
    for region in to_delete: del props_dict[region]
    return regions_found

def classification(regions_found, props_dict):
    spores = np.zeros(regions_found.shape)
    hifas = np.zeros(regions_found.shape)
    spore_count = 0
    hifas_count = 0
    hifas_labels = []

    for region in props_dict.keys():
        rate = props_dict[region]["area"] / props_dict[region]["perimeter"]
        if np.mean(props_dict[region]["eccentricity"]) > 0.7 and props_dict[region]["area"] > 150000:
            hifas[regions_found == region] = 1
            hifas_count += 1
            hifas_labels.append(region)
        else:
            spores[regions_found == region] = 1
            spore_count += 1
    print(f"Hay {spore_count} esporas y {hifas_count} esporas con hifas")
    return spores, hifas, hifas_labels

def colorize_image(segmentated, regions_found, path):
    for h in range(regions_found.shape[0]):
        mask = segmentated[h]
        img = regions_found[h].astype(np.int64)
        colorized = label2rgb(img, image=mask, bg_label=0)
        colorized = equalize(colorized).astype(np.uint8)
        save_rgb_image(colorized, f'{PATH}{path}_{h}.png')

def skeletonize_image(image):
    skeleton = skeletonize(image)
    # save_image(skeleton_label*255, f'results/hifas.png')
    return skeleton

def get_length(image):
    skeleton = skeletonize_image(image)
    total_area = 0
    for region in regionprops(skeleton.astype(np.uint8)):
        total_area += region.perimeter
    return total_area

def separate_hifas_length(hifas, hifas_labels):
    length_dict = {label : 0 for label in hifas_labels}
    for hifa_label in hifas_labels:
        for h in range(len(hifas)):
            hifa = np.zeros(hifas[h].shape)
            hifa[hifas[h] == hifa_label] = 1
            length = get_length(hifa) * 0.12
            if length > length_dict[hifa_label]:
                length_dict[hifa_label] = int(length) 
    return length_dict


def segmentate_matrix(matrix, prefix):
    segmentated = []
    for i in range(matrix.shape[0]):
        segmentated.append(segmentation(i, matrix, prefix))
    regions_found = region_growing3D(np.array(segmentated))
    props_dict = get_props_per_region(regions_found)
    regions_found = remove_noise(regions_found, props_dict)
    spores, hifas, hifas_labels = classification(regions_found, props_dict)
    contours = ske.get_contours(regions_found, hifas_labels, prefix, save_image, equalize) 
    # colorize_image(segmentated, spores, "high_spores")
    # colorize_image(segmentated, hifas, "high_hifas")