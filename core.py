#encoding: utf-8
import sys
import os
import os.path
import cv2
import numpy as np


def equlizehist(image_path, denoise=False, verbose=False):
    bgr = cv2.imread(image_path)
    bgr[:, :, 0] = cv2.equalizeHist(bgr[:, :, 0])
    bgr[:, :, 1] = cv2.equalizeHist(bgr[:, :, 1])
    bgr[:, :, 2] = cv2.equalizeHist(bgr[:, :, 2])

    if denoise:
        # bgr = cv2.fastNlMeansDenoisingColoredMulti([bgr, bgr, bgr, bgr, bgr], 2, 5, None, 4, 5, 35)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def clahe_hsv(image_path, verbose=False):
    bgr = cv2.imread(image_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=4)
    hsv_planes[0] = clahe.apply(hsv_planes[2])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(hsv_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def clahe(image_path, denoise=False, verbose=False):
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4)
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        #bgr = cv2.fastNlMeansDenoisingColoredMulti([bgr, bgr, bgr, bgr, bgr], 2, 5, None, 4, 5, 35)
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def convert(image_dir, output_home, option="clahe"):
    list = os.listdir(image_dir)
    for f in list:
        if "jpg" in f:
            print("processing: %s" % (f))
            image_path = os.path.join(image_dir, f)
            if option == "clahe":
                bgr = clahe(image_path)
            elif option == "clahe_de":
                bgr = clahe(image_path, denoise=True)
            elif option == "clahe_hsv":
                bgr = clahe_hsv(image_path)
            elif option == "eql_de":
                bgr = equlizehist(image_path, denoise=True)
            else:
                bgr = equlizehist(image_path)
            basename = os.path.basename(image_dir)
            output_dir = os.path.join(output_home, basename)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_file = os.path.join(output_dir, f)
            cv2.imwrite(output_file, bgr)

if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)

    if argc < 3:
        print("[Usage]python %s <option clahe, clahe_de, clahe_hsv, eql, eql_de> <image home dir> <output home dir>" % (argvs[0]))
        sys.exit(-1)

    option = argvs[1]
    image_home_dir = argvs[2]
    output_home_dir = argvs[3]

    list_dir = os.listdir(image_home_dir)

    for dir in list_dir:
        image_dir_path = os.path.join(image_home_dir, dir)
        if not os.path.isdir(image_dir_path):
            continue
        convert(image_dir_path, output_home_dir, option=option)

    # equlizehist()
    # clahe()