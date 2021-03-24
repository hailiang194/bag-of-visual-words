import cv2
import config
import sys
from scipy.cluster.vq import vq 
import numpy as np

svm_cost_function = lambda X, y, weight: y * np.amax(np.dstack((np.zeros(y.shape), 1 - weight @ X.T)), axis=0) + (1 - y) * np.amax(np.dstack((np.zeros(y.shape), 1 + weight @ X.T)), axis=0) 

def get_word(image, detector, codebook):
    keypoint, desc = detector.detectAndCompute(image, None)
    words, _ = vq(desc, codebook)

    return keypoint, desc, words

if __name__ == "__main__":
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    codebook = np.loadtxt("./codebook.txt")
    weight = np.loadtxt("./weight.txt")
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    kp, desc, words = get_word(image, config.detector, codebook)
    
    # grid_shape = (5, 5, codebook.shape[0])
    # grid_hist = np.zeros(grid_shape)
    # grid_size = np.ceil(np.array(list(image.shape[:2])) / np.array(list(grid_shape[:2]))).astype(int)
     
    # for i in range(len(kp)):
    #     x = int(kp[i].pt[0] // grid_size[1])
    #     y = int(kp[i].pt[1] // grid_size[0])
    #     grid_hist[x, y, int(words[i])] += 1

    # test = np.dstack([image] * 3)
    # for x in range(grid_shape[1]):
    #     for y in range(grid_shape[0]):
    #         # cv2.line(test, (x * grid_size[1], y * grid_size[0]), ((x + 1) * grid_size[1], (y + 1) * grid_size[0]), (0, 0, 255), 6)
    #         is_book = ((weight @ grid_hist[x, y].T) >= 0).astype(int)
    #         # print(np.array([is_book]).shape)
    #         # coffident = np.abs(svm_cost_function(grid_hist[x, y], np.array([is_book]), weight))[0, 1]
    #         # if coffident < 0.2:
    #         #     break
    #         # print(svm_cost_function(grid_hist[x, y], , weight))
    #         if is_book:
    #             cv2.rectangle(test, (x * grid_size[1], y * grid_size[0]), ((x + 1) * grid_size[1] - 3, (y + 1) * grid_size[0] - 3), (0, 255, 0), 3)
    #         else:
    #             cv2.rectangle(test, (x * grid_size[1], y * grid_size[0]), ((x + 1) * grid_size[1] - 3, (y + 1) * grid_size[0] - 3), (255, 0, 0), 3)

    # cv2.imshow("Test", test)
    # cv2.waitKey(0)
    
    

