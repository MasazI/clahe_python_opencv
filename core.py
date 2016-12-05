#encoding: utf-8
import cv2


def equlizehist():
    bgr = cv2.imread("/Users/masai/Desktop/Customers/TOYOTA/20161216_first_milestone_testset/toyota_testsets_dec_20161202/ID8776/121_121.jpg")
    bgr[:, :, 0] = cv2.equalizeHist(bgr[:, :, 0])
    bgr[:, :, 1] = cv2.equalizeHist(bgr[:, :, 1])
    bgr[:, :, 2] = cv2.equalizeHist(bgr[:, :, 2])
    cv2.imshow("test", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clahe():
    bgr = cv2.imread("/Users/masai/Desktop/Customers/TOYOTA/20161216_first_milestone_testset/toyota_testsets_dec_20161202/ID8776/121_121.jpg")
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4)
    lab_planes[0] = clahe.apply(lab_planes[0])
    #lab_planes[1] = clahe.apply(lab_planes[1])
    #lab_planes[2] = clahe.apply(lab_planes[2])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imshow("test", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    equlizehist()
    clahe()