import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
from tabulate import tabulate

def main():
    pic = mpimg.imread("hopping-street.jpg")
    newPic = [[0] * 3 for i in range(3)]
    clusterList = [2, 5, 10, 25, 50, 100, 200, 1000]
    errorList = []
    row_number = 103
    col_number = 131
    h = 309
    w = 393

    block_row = np.array_split(pic, row_number, axis=0)
    img_blocks = []
    for block in block_row:
        block_col = np.array_split(block, col_number, axis=1)
        img_blocks += [block_col]

    pixel_1 = []
    for i in range(103):
        for j in range(131):
            pixel_1.append(np.concatenate(img_blocks[i][j][0:3], axis=None))
    pixel = np.stack(pixel_1, axis=0)

    for cl in clusterList:
        # print("current cluster number is:", cl)
        kmeans = KMeans(n_clusters=cl, random_state=0).fit(pixel)
        cluster_assignments = kmeans.predict(pixel)
        cluster_centers = kmeans.cluster_centers_
        compressed_img = np.zeros((h, w, 3), dtype=np.uint8)
        pixel_count = 0
        for i in range(103):
            for j in range(131):
                cluster_idx = cluster_assignments[pixel_count]
                cluster_value = cluster_centers[cluster_idx]
                cluster = cluster_value.reshape(3, 3, 3)
                for k in range(3):
                    compressed_img[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = cluster[k]
                pixel_count += 1
        err = np.sum((pic.astype("float") - compressed_img.astype("float")) ** 2)
        err /= float(pic.shape[0] * pic.shape[1])
        errorList.append(err)
        err = 0

    info = {'cluster number': clusterList, 'reconstruction error': errorList}
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))


if __name__ == "__main__":
        main()