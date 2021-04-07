import predict
import cv2
import sys
import numpy as np
import config

if __name__ == "__main__":
    code_book = np.loadtxt("./codebook.txt")
    weight = np.loadtxt("./weight.txt")

    database_vector = np.loadtxt("./query_vector.txt")
    image = cv2.imread(sys.argv[1])
    matching_images = predict.get_matching_image(image, code_book, config.detector, weight, database_vector)
    for path in matching_images.keys():
        print("file://P:/opencv/" + config.paths[0][3:] + path.strip(), matching_images[path])