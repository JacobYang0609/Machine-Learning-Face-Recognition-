import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def main():
    imgs = []
    pixel_1 = []
    for i in range(100):
        imgs.append(mpimg.imread('Faces/face_{}.png'.format(i), 1))
    imgs = np.asarray(imgs)
    pixel = imgs.reshape(100, 2500)
    mean = np.mean(pixel, axis=0)
    new_p = pixel - mean
    cov = np.dot(np.transpose(new_p), new_p)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_inx = np.argsort(-eig_val)
    Ks = [3, 5, 10, 30, 50, 100]

    # The follow for-loop shows the compressed imgae for face_9
    for k in Ks:
        w = eig_vec[:, eig_inx[:k]]
        y = pixel[9].dot(w)
        x = np.dot(y, w.T)
        x_new = np.array(x, dtype=np.float64)
        x_n = x_new.reshape(50, 50)
        print("current k number is:", k)
        plt.imshow(x_n)
        plt.show()

if __name__ == "__main__":
        main()