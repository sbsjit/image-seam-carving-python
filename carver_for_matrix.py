from re import S
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
    # imwrite(energy_map_filename, energy_map)

    return energy_map

def crop_c(scale_c):

    # Random image generator
    img_array = np.random.rand(10,10,3) * 255

    r, c, _ = img_array.shape
    print("Pixels per row(height) ::", r)
    print("Pixels per column(width) ::", c)
    print("Number of colors/channels ::", _) #RGB

    new_c = int(scale_c * c)

    # trange from tqdm displays the smart progress meter of the loop
    # for i in trange(c - new_c):
    #     img = carve_column(img_array)
    # return img

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

    j = np.argmin(M[-1]) # j -> individual column number with minimum energy value from all of the last row energies 
    for i in reversed(range(r)): # loop starts from i, i-1, ... ..., 0
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    # print(mask)

    # name = "./seam-images/" + sys.argv[3] + str(name_id) + ".jpg"
    # Visualizing the pixel seams of the input image as an actual image
    # imwrite(name, img) 
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    # energy_map = calc_energy(img)
    # print("Energy Map:: ", energy_map)
    energy_map = np.random.randint(0, 100, size=(30, 10, 2))
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]

            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
            # print(min_energy)
            # print(backtrack[i, j])
    
    return M, backtrack

def main():
    if len(sys.argv) != 4:
        print('usage: carver.py <row/column> <scale> <input_energy_map>', file=sys.stderr)
        sys.exit(1)

    seam_axis = sys.argv[1]
    scale = float(sys.argv[2])
    energy_map_output = sys.argv[3]

    if seam_axis == 'column':
        output_image = crop_c(scale)
    elif seam_axis == 'row':
        output_image = crop_r(scale)
    else:
        print('usage: carver.py <row/column> <scale> <input_energy_map>', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()