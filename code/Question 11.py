import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
from tabulate import tabulate

def main():
    clusterList = [2, 5, 10, 25, 50, 100, 200, 1000]
    totalNumbers = [121485, 282088, 403645, 564536, 686573, 809210, 933047, 1234215]
    info = {'K value': clusterList, 'total numbers': totalNumbers}
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

if __name__ == "__main__":
        main()