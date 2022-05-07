import sys

from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from distutils.log import debug

name_id = 0

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)
    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)
    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)
    # Visualizing the energy map of the input image as an actual image
    # energy_map_filename = sys.argv[5]
    # imwrite(energy_map_filename, (energy_map))
    return energy_map

def crop_c(img, scale_c):
    r, c, _ = img.shape

    print("Pixels per row(height) ::", r)
    print("Pixels per column(width) ::", c)
    print("Number of colors/channels ::", _) #RGB
    new_c = int(scale_c * c)

    # trange from tqdm displays the smart progress meter of the loop
    for i in trange(c - new_c):
        img = carve_column(img)
    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):
    r, c, _ = img.shape
    global name_id
    name_id += 1
    M, backtrack = minimum_seam(img)
    
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in reversed(range(r)): # loop starts from m-1, ... ..., 0
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)

    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            elif j == c - 1:
                idx = np.argmin(M[i - 1, j - 1:j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1] 
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    
    return M, backtrack

def main():
    if len(sys.argv) != 6:
        print('usage: carver.py <row/column> <scale> <image_input> <image_output> <input_energy_map>', file=sys.stderr)
        sys.exit(1)

    seam_axis = sys.argv[1]
    scale = float(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    img = imread(input_file)

    if seam_axis == 'column':
        output_image = crop_c(img, scale)
    elif seam_axis == 'row':
        output_image = crop_r(img, scale)
    else:
        print('usage: carver.py <row/column> <scale> <image_input> <image_output> <input_energy_map>', file=sys.stderr)
        sys.exit(1)
    
    imwrite(output_file, output_image)

if __name__ == '__main__':
    main()