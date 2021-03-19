import cv2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.cluster.vq import vq
import sys
import numpy as np
import config
import csv

def get_predict_value(image, detector, codebook, svm_weight):
    _, desc = detector.detectAndCompute(image, None) 
    words, _ = vq(desc, codebook)
    hist = np.zeros((1, codebook.shape[0]))
    for word in words.tolist():
        hist[0, word] += 1

    
    return int(svm_weight @ hist.T > 0) 

def test(path, detector, codebook, weight, expValue):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    predicted = get_predict_value(image, detector, codebook, weight)

    return [path, predicted, expValue, predicted == expValue]

if __name__ == "__main__":
    codebook = np.loadtxt("./codebook.txt")
    weight = np.loadtxt("./weight.txt")
    label = {}
    with open("./labels.txt") as label_reader:
        for line in label_reader.readlines():
            token = (line.split())
            label[token[0]] = int(token[1])
    
    test_case = [
    test("../another-half-xiao-mi.png", config.detector, codebook, weight, label["../backgound/"]),
    test("../box.png", config.detector, codebook, weight, label["../backgound/"]),
    test("../genm.jpg", config.detector, codebook, weight, label["../backgound/"]),
    test("../half-chua-ruoi.png", config.detector, codebook, weight, label["../books/"]),
    test("../half-xiaomi.png", config.detector, codebook, weight, label["../backgound/"]),
    test("../IMG_20210310_141623.jpg", config.detector, codebook, weight, label["../books/"]),
    test("../meo.jpg", config.detector, codebook, weight, label["../books/"]),
    test("../meow.jpg", config.detector, codebook, weight, label["../books/"]),
    test("../notebook.png", config.detector, codebook, weight, label["../backgound/"]),
    test("../object.png", config.detector, codebook, weight, label["../books/"]),
    test("../object-box.png", config.detector, codebook, weight, label["../backgound/"]),
    test("../object-xiaomi.png", config.detector, codebook, weight, label["../backgound/"]),
    test("../review-chua-ruoi-anh-chup.png", config.detector, codebook, weight, label["../books/"]),
    test("../sample_image.jpeg", config.detector, codebook, weight, label["../books/"]),
    test("../toi-la-ai-–-va-neu-vay-thi-bao-nhieu-.jpg", config.detector, codebook, weight, label["../books/"]),
    test("../watermelon-gun.jpg", config.detector, codebook, weight, label["../backgound/"]),
    test("../xiaomi.png", config.detector, codebook, weight, label["../backgound/"]),
    ]

    with open("report.csv", "w") as report:
        csv_writer = csv.writer(report)
        csv_writer.writerow(['Path', 'Predict', 'Expected value', 'Result'])
        csv_writer.writerows(test_case)
