import numpy as np
from tabulate import tabulate
#If the original images are 64-bit float, the compression rate function will be:
def main():

    k = [3, 5, 10, 30, 50, 100]
    log = np.log2(k)
    pixel = 50 * 50
    CR = (log * pixel + np.dot(k, 64)) / (64 * pixel)
    info = {'K': k, 'CR (original images 64-bit float)': CR}
    print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

if __name__ == "__main__":
        main()