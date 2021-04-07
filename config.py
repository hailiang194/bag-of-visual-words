import cv2
# import rootsift

detector = cv2.SIFT_create()
# detector = cv2.ORB_create()
paths = ["../ukbench/ukbench/full/"]

def pre_process_image(image):
    process = image.copy()
    if len(image.shape) == 3 and image.shape[-1] == 3:
        process = cv2.cvtColor(process, cv2.COLOR_BGRA2GRAY)

    # process = cv2.GaussianBlur(process, (7, 7), 0)
    return process
