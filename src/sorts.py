import cv2


def sort_second(val):
    return val[1]


def sort_by_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm
