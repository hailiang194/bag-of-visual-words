import cv2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.cluster.vq import vq
import sys
import numpy as np
import config
import cv2
import config
import numpy as np
from scipy.cluster.vq import vq
import sys
import os
import imutils

def get_image_hist(image, detector, code_book):
    clone_image = image.copy()
    if len(image.shape) > 2 and image.shape[-1] == 3:
        clone_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    clone_image = config.pre_process_image(clone_image)
    _, desc = detector.detectAndCompute(clone_image, None)
    labels, _ = vq(desc, code_book) 
    
    hist = np.zeros((1, code_book.shape[0]))
    for label in labels.tolist():
        hist[0, label] += 1

    hist = hist / np.linalg.norm(hist)
    return hist


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1]) 
    image = imutils.resize(image, height=475)
    cv2.imshow("Query", image)
    code_book = np.loadtxt("./codebook.txt")
    weight = np.loadtxt("./weight.txt")
    hist = get_image_hist(image, config.detector,code_book)
    vector = np.multiply(hist, weight).reshape(code_book.shape[0])
    distance_vector = np.linalg.norm(vector)
    database_vector = np.loadtxt("./query_vector.txt")
    # print(database_vector.shape)
    score = np.zeros((1, database_vector.shape[0]))
    for i in range(database_vector.shape[0]):
        data = database_vector[i, :]
        distance_data = np.linalg.norm(data)
        score[0, i] = np.linalg.norm(data / distance_data - vector / distance_vector)

    accepted_score = np.sort(score)[0, 10]
    for i, path in enumerate(os.listdir("../books/")):
        if score[0, i] <= accepted_score:
            expected = " Expected" if "../books/" + path in sys.argv[2:]  else ""
            print(str(score[0, i]) + expected)
            mapping_image = cv2.imread("../books/" + path)
            cv2.imshow("Mapping", mapping_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
