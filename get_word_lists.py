import numpy as np
import sys
import cv2
import config
from scipy.cluster.vq import vq
import imutils

if __name__ == "__main__":
    codebook = np.loadtxt("./codebook.txt")
    image_list = np.loadtxt("./all_hist.txt")
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    image = imutils.resize(image, height=475) 
    path = []
    with open("./image_path.txt", "r") as file:
        path.extend(file.readlines())

    keypoint, desc = config.detector.detectAndCompute(image, None)
    words, _ = vq(desc, codebook)

    hist = np.zeros((1, codebook.shape[0]))

    for word in words:
        hist[0, word] += 1
    
    hist = hist / np.max(hist)
    # hist = hist / 4065.9280613409774

    print(hist)
    distance = []
    print(image_list.shape)
    # print(np.sum(image_list))
    for i in range(image_list.shape[0]):
        dist = np.linalg.norm(hist - image_list[i, :])
        distance.append(dist)

    min_value = sorted(distance)[20]
    # print(sorted(distance))
    cv2.imshow("Image", image)
    for i in range(image_list.shape[0]):
        # print(distance[i])
        if distance[i] <= min_value:
            print(path[i])
            query = cv2.imread(path[i].strip(), cv2.IMREAD_GRAYSCALE)
            cv2.imshow("Query", query)
            cv2.waitKey(0)
            # cv2.imshow("query", cv2.imread(path[i], cv2.IMREAD_GRAYSCALE))
            # cv2.waitKey(0)
