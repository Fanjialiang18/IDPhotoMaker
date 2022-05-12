from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import sys


def _help():
    print("Usage:")
    print("     python face_align.py <path of a picture>")
    print("For example:")
    print("     python face_align.py pic/HL.jpg")


def face_align(img_path,save_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")

    # 初始化 FaceAligner 类对象
    fa = FaceAligner(predictor, desiredFaceWidth=358, desiredFaceHeight=443,
                     desiredLeftEye=(0.38, 0.38))

    image = cv2.imread(img_path)
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Input", image)
    rects = detector(gray, 2)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)
        # 证件照大小 358*443像素
        # 调用 align 函数对图像中的指定人脸进行处理
        face_aligned = fa.align(image, gray, rect)
        cv2.imshow("Original", face_orig)
        cv2.imshow("Aligned", face_aligned)
        cv2.imwrite(save_path,face_aligned,[int( cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.waitKey(5000)

    # if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
    #     _help()
    # else:
    # face_align(sys.argv[1])


if __name__ == "__main__":
    # face_align(img_path="input_phone/1.jpg",save_path="align/1.jpg")

    face_align(img_path="output_phone/222_fg.jpg",save_path="align/3.jpg")
    # face_align(img_path="output_phone/2_fg.jpg")
