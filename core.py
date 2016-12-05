#encoding: utf-8
import sys
import os
import os.path
import cv2


def equlizehist(image_path, verbose=False):
    bgr = cv2.imread(image_path)
    bgr[:, :, 0] = cv2.equalizeHist(bgr[:, :, 0])
    bgr[:, :, 1] = cv2.equalizeHist(bgr[:, :, 1])
    bgr[:, :, 2] = cv2.equalizeHist(bgr[:, :, 2])
    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def clahe(image_path, verbose=False):
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4)
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def convert(image_dir, output_home, option="clahe"):
    list = os.listdir(image_dir)
    for f in list:
        if "jpg" in f:
            image_path = os.path.join(image_dir, f)
            if option == "clahe":
                bgr = clahe(image_path)
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
        print("[Usage]python %s <option clahe or eql> <image dir> <output home dir>" % (argvs[0]))
        sys.exit(-1)

    option = argvs[1]
    image_dir = argvs[2]
    output_home_dir = argvs[3]

    convert(image_dir, output_home_dir, option=option)

    # equlizehist()
    # clahe()