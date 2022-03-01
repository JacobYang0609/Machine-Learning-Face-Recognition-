import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def main():
    img = mpimg.imread('face.png')[:, :, 0]
    Ks = [3, 5, 10, 30, 50, 100]
    pca = PCA()
    pca.fit(img)
    cov = pca.get_covariance()
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(50)]
    eig_pairs.sort(reverse=True)
    for k in Ks:
        w = np.array([ele[1] for ele in eig_pairs[:k]])
        y = np.dot(img, np.transpose(w))
        x_new = np.dot(y, w)
        print("current k number is:", k)
        plt.imshow(x_new)
        plt.show()

if __name__ == "__main__":
        main()