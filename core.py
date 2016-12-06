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


def clahe_inv(image_path, denoise=False, verbose=False, limit=None):
    bgr = cv2.imread(image_path)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        # bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    # lab_planes[1] = clahe.apply(lab_planes[1])
    # lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def clahe(image_path, denoise=False, verbose=False, limit=None):
    bgr = cv2.imread(image_path)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        #bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr

def b_clahe_nlm(image_path, denoise=False, verbose=False, limit=None):
    bgr = cv2.imread(image_path)

    if denoise:
        bgr = cv2.bilateralFilter(bgr, 3, 3, 2)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 3, 9)
        #bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr



def clahe_inv_b(image_path, denoise=False, verbose=False, limit=None):
    bgr = cv2.imread(image_path)

    if denoise:
        bgr = cv2.bilateralFilter(bgr, 15, 5, 2)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    # lab_planes[1] = clahe.apply(lab_planes[1])
    # lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def clahe_b(image_path, denoise=False, verbose=False, limit=None):
    bgr = cv2.imread(image_path)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit)
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.bilateralFilter(bgr, 15, 5, 2)

    if verbose:
        cv2.imshow("test", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bgr


def convert(image_dir, output_home, option="clahe", limit=None, verbose=False):
    list = os.listdir(image_dir)
    for f in list:
        if "jpg" in f:
            print("processing: %s" % (f))
            image_path = os.path.join(image_dir, f)
            if option == "clahe":
                bgr = clahe(image_path, limit=limit)
            elif option == "clahe_de":
                bgr = clahe(image_path, denoise=True, limit=limit)
            elif option == "clahe_hsv":
                bgr = clahe_hsv(image_path)
            elif option == "eql_de":
                bgr = equlizehist(image_path, denoise=True)
            elif option == "best":
                bgr = b_clahe_nlm(image_path, denoise=True, limit=limit)
            elif option == "all":
                bgr = b_clahe_nlm(image_path, denoise=True, limit=limit)
                bgr_b = clahe_b(image_path, denoise=True, limit=limit)
                bgr_inv = clahe_inv(image_path, denoise=True, limit=limit)
                bgr_inv_b = clahe_inv_b(image_path, denoise=True, limit=limit)

                bgr_2 = clahe(image_path, denoise=True, limit=2)
                bgr_b_2 = clahe_b(image_path, denoise=True, limit=2)
                bgr_inv_2 = clahe_inv(image_path, denoise=True, limit=2)
                bgr_inv_b_2 = clahe_inv_b(image_path, denoise=True, limit=2)

                vis = np.concatenate((bgr, bgr_b, bgr_inv, bgr_inv_b), axis=1)
                vis_2 = np.concatenate((bgr_2, bgr_b_2, bgr_inv_2, bgr_inv_b_2), axis=1)

                v = np.concatenate((vis, vis_2), axis=0)

                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(v, "CLAHE_NLM", (10, 100), font, 3, (0, 0, 255))
                cv2.putText(v, "CLAHE_BI", (650, 100), font, 3, (0, 0, 255))
                cv2.putText(v, "NLM_CLAHE", (1300, 100), font, 3, (0, 0, 255))
                cv2.putText(v, "BI_CLAHE", (1950, 100), font, 3, (0, 0, 255))

                cv2.putText(v, "CLAHE_NLM_2", (10, 590), font, 3, (0, 0, 255))
                cv2.putText(v, "CLAHE_2_BI", (650, 590), font, 3, (0, 0, 255))
                cv2.putText(v, "NLM_CLAHE_2", (1300, 590), font, 3, (0, 0, 255))
                cv2.putText(v, "BI_CLAHE_2", (1950, 590), font, 3, (0, 0, 255))

                basename = os.path.basename(image_dir)
                output_dir = os.path.join(output_home, basename)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                output_file = os.path.join(output_dir, f)
                print output_file
                cv2.imwrite(output_file, v)

            else:
                bgr = equlizehist(image_path)

            if not option == "all":
                basename = os.path.basename(image_dir)
                output_dir = os.path.join(output_home, basename)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                output_file = os.path.join(output_dir, f)
                cv2.imwrite(output_file, bgr)

if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)

    if argc < 4:
        print("[Usage]python %s <option clahe, clahe_de, clahe_hsv, eql, eql_de, best, all> <image home dir> <output home dir> <limit>" % (argvs[0]))
        sys.exit(-1)

    option = argvs[1]
    image_home_dir = argvs[2]
    output_home_dir = argvs[3]
    limit = argvs[4]
    list_dir = os.listdir(image_home_dir)
    for dir in list_dir:
        image_dir_path = os.path.join(image_home_dir, dir)
        if not os.path.isdir(image_dir_path):
            continue
        convert(image_dir_path, output_home_dir, option=option, limit=int(limit), verbose=True)

    # equlizehist()
    # clahe()