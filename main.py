


import numpy as np
import cv2 as cv



import math
def non():
    font = cv.FONT_HERSHEY_SIMPLEX

    #print("Enter image name:")
    #img_name = input()
    #img_name = 'scan0008_reverse'
    #print(f"Selected image: {img_name}.tif")
    #img = cv2.imread(img_name+".tif")
    #img_copy = img.copy()
    img1 = cv.imread("scan0001_rot.tif")
    img2 = cv.imread("scan0002_rot.tif")
    img3 = cv.imread("scan0003_rot.tif")

    img1 = cv.GaussianBlur(img1, ksize=(9, 9), sigmaX =0, sigmaY=0)
    img2 = cv.GaussianBlur(img2, ksize=(9, 9), sigmaX =0, sigmaY=0)
    img3 = cv.GaussianBlur(img2, ksize=(9, 9), sigmaX =0, sigmaY=0)

    img1 = np.uint16(img1)
    img2 = np.uint16(img2)
    img3 = np.uint16(img3)

    res = np.uint8((img1+img2+img3)//3)

    #r = np.multiply(r,res)
    cv.imshow("red", res)
    cv.imwrite("scan000_gauss9.tif", res)
    cv.waitKey(0)
    cv.destroyAllWindows()


# на вход подать изображение и область (распознанную) по которой обрезать
def crop_rot_rect(img, rect):
    # rect ->  ((центр прямогуольника), (размер прямоугольника), угол наклона)
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)    # формирует матрицу поворота
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))        # вращает изображение

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop


def HSV_analyzer():
    def nothing(*arg):
        pass

    # имя файла, который будем анализировать
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/22.02.2023/res/overlaped.tif'
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/scan0001.tif'
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/1.jpg'

    img = cv.imread(fn)
   # img = cv.medianBlur(img, 21)

    #cv2.imshow(img)
    print(img.shape)
    y_range, x_range, _ = img.shape  # задаем рамзеры картинки

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('result', int(x_range // 2), int(y_range // 2))  # уменьшаем картинку в 3 раза
    cv.namedWindow("settings")  # создаем окно настроек

    cap = img

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv.createTrackbar('h1', 'settings', 0, 255, nothing)
    cv.createTrackbar('s1', 'settings', 0, 255, nothing)
    cv.createTrackbar('v1', 'settings', 0, 255, nothing)
    cv.createTrackbar('h2', 'settings', 255, 255, nothing)
    cv.createTrackbar('s2', 'settings', 255, 255, nothing)
    cv.createTrackbar('v2', 'settings', 255, 255, nothing)
    crange = [0, 0, 0, 0, 0, 0]
    cv.resizeWindow('settings', 700, 140)

    while True:
        #######flag, img = cap.read()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv.getTrackbarPos('h1', 'settings')
        s1 = cv.getTrackbarPos('s1', 'settings')
        v1 = cv.getTrackbarPos('v1', 'settings')
        h2 = cv.getTrackbarPos('h2', 'settings')
        s2 = cv.getTrackbarPos('s2', 'settings')
        v2 = cv.getTrackbarPos('v2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        thresh = cv.inRange(hsv, h_min, h_max)


        #blur = cv2.medianBlur(thresh, 21)
       # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
        cv.imshow('result', thresh)

        ch = cv.waitKey(5)
        if ch == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def binarization_analyzer():
    def nothing(*arg):
        pass

    # имя файла, который будем анализировать
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/scan0001.tif'
    #fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/1.jpg'

    img = cv.imread(fn)

    #img = cv.GaussianBlur(img, (25, 25), 9)
    #img = cv2.medianBlur(img, 21)

    y_range, x_range, _ = img.shape

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('result', int(x_range // 2), int(y_range // 2))  # уменьшаем картинку в 3 раза
    cv.namedWindow("settings")  # создаем окно настроек

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv.createTrackbar('low', 'settings', 0, 255, nothing)
    cv.createTrackbar('high', 'settings', 0, 255, nothing)

    cv.resizeWindow('settings', 700, 140)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    while True:
        #######flag, img = cap.read()
        # считываем значения бегунков
        h1 = cv.getTrackbarPos('low', 'settings')
        s1 = cv.getTrackbarPos('high', 'settings')

        #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 1001, 0 + h1 // 10)
        #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 501, 0 + h1 // 10)

        # ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_BINARY)
        #
        # kernel = np.ones((25, 25), dtype=np.uint8)
        # thresh = cv.erode(thresh, kernel)
        # thresh = cv2.dilate(thresh, kernel)

        ret, thresh = cv.threshold(img, h1, s1, cv.THRESH_BINARY)

        # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
        cv.imshow('result', thresh)

        ch = cv.waitKey(5)
        if ch == 27:
            break

    cv.destroyAllWindows()


def contours_finder():
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/1.jpg'
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/scan0001.tif'
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/binar.tif'
    img = cv.imread(fn)

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    y_range, x_range, _ = img.shape  # задаем рамзеры картинки
    cv.resizeWindow('result', int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_img, 135, 255, cv.THRESH_BINARY)

    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_lst = []

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат

        area = int(rect[1][0] * rect[1][1])  # вычисление площади

        if area > 3:
            contours_lst.append(area)

        if area > 170:  # если площадь прямогульника больше  < 9_000_000
            #print(f'rect params: {rect}')
           # print(area)  # примерно должно быть 6_500_500
            cv.drawContours(img, [box], 0, (0, 0, 255), 2)      # отрисовка прямогуольников размером больше 700_000

            color_yellow = (0, 255, 255)
            center = (int(rect[0][0]), int(rect[0][1]))
            cv.putText(img, f'{area}', (center[0] + 20, center[1] - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)

    cv.imwrite('C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/res.tif', img)
    cv.imshow('result', img)
    cv.imwrite('C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/res.tif', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    contours_lst.sort()
    print(contours_lst)


def binaryze(img_path, threshold=100):
    def nothing(*arg):
        pass

    image = cv.imread(img_path)

    y_range, x_range, _ = image.shape
    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('result', int(x_range // 2), int(y_range // 2))  # уменьшаем картинку в 3 раза

    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)

    cv.imshow('result', thresh)

    cv.waitKey(0)

    cv.destroyAllWindows()
    return thresh





# на вход подается бинаризированное изображение
# на выходе координаты контуров

def find_contours(bin_image, more_than=0):
    contours0, hierarchy = cv.findContours(bin_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        area = int(rect[1][0] * rect[1][1])  # вычисление площади прямоугольника
        if area > more_than:
            rectangles.append(rect)

    rectangles.sort()
    del rectangles[-1] # удаляем максимальный прямогуольник
    return rectangles


def crop_image(img, rect):
    # на вход подать изображение и область (распознанную) по которой обрезать

    # rect ->  ((центр прямогуольника), (размер прямоугольника), угол наклона)
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)  # формирует матрицу поворота
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))  # вращает изображение

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop


if __name__ == '__main__':
    #HSV_analyzer()
    #contours_finder()
    #binarization()
    #binarization_analyzer()
    #contours_finder()

    path = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/scan0001.tif'
    bin_img = binaryze(path, threshold=220)
    #cv.imwrite('C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/binar.tif', bin_img)
    print(find_contours(bin_img, more_than=6_000))

    # 1) бинаризовать изображение и обрезать изображения фоторезиста просто от белого листа
    # 2) уже на вырезанных иозображениях искать контуры
    # 3) по найденным контурам поворачивать изображения и сохранять результат