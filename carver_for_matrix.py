import numpy as np


def minimum_seam():

    energy_map = [[0, 20, 40, 10, 30], [0, 20, 90,15,120], [80,100,10,250,80], [150,50,200,10,300], [35,70,40,5,80]]

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, 4):
        for j in range(0, 4):
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]

            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    
    return M, backtrack

def main():
    print(minimum_seam())

if __name__ == '__main__':
    main()