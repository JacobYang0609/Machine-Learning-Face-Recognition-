import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
from tabulate import tabulate

def main():
    clusterList = [2, 5, 10, 25, 50, 100, 200, 1000]
    compressionRate = [0.04, 0.1, 0.14, 0.19, 0.24, 0.28, 0.32, 0.42]
    info = {'K value': clusterList, 'compression rate': compressionRate}
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

if __name__ == "__main__":
        main()