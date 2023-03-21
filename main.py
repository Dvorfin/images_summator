import numpy as np
import cv2 as cv

import math


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
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/1.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/scan0001.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/brightness_32/scan0001.tif'

    img = cv.imread(fn)

    #img = cv.GaussianBlur(img, (25, 25), 9)
    #img = cv.medianBlur(img, 21)

    y_range, x_range, _ = img.shape

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('result', int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза
    cv.namedWindow("settings")  # создаем окно настроек

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv.createTrackbar('thresh', 'settings', 134, 255, nothing)
    cv.createTrackbar('high', 'settings', 0, 1000, nothing)

    cv.resizeWindow('settings', 700, 140)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    while True:
        img = cv.imread(fn)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #######flag, img = cap.read()
        # считываем значения бегунков
        h1 = cv.getTrackbarPos('thresh', 'settings')
        s1 = cv.getTrackbarPos('high', 'settings')

        #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 1001, 0 + h1 // 10)
        #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 501, 0 + h1 // 10)

        # ret, thresh = cv2.threshold(img, h1, s1, cv2.THRESH_BINARY)
        #
        # kernel = np.ones((25, 25), dtype=np.uint8)
        # thresh = cv.erode(thresh, kernel)
        # thresh = cv2.dilate(thresh, kernel)

        ret, thresh = cv.threshold(img, h1, 255, cv.THRESH_BINARY)
        #ret, thresh = cv.threshold(img, h1, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        contours0, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        for cnt in contours0:
            rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            area = int(rect[1][0] * rect[1][1])  # вычисление площади прямоугольника
            box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
            box = np.int0(box)  # округление координат
            #print(area)
            #if area > s1*10:
            if 90000 < area < 33227154:  # если площадь прямогульника больше  < 9_000_000
            #if 100 < area < 400:
                print(f'area params: {area}')
                cv.drawContours(img, [box], 0, (0, 0, 255), 8)  # отрисовка прямогуольников размером больше 700_000
                fr = 'C:/Users/vadik/Desktop/STUDY/diplom/scans/10.03.2023/boba.tif'
                cv.imwrite(fr, img)
                # color_yellow = (0, 255, 255)
                # center = (int(rect[0][0]), int(rect[0][1]))
                # cv.putText(img, f'{area}', (center[0] + 20, center[1] - 20),
                #            cv.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)






        # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
            cv.imshow('result', img)
       # break
        ch = cv.waitKey(5)
        if ch == 27:
            break

    cv.destroyAllWindows()


def contours_finder():
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/1.jpg'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/crop.tif'
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/scan0006.tif'
    #fn = 'C:/Users/vadik/Desktop/STUDY/diplom/10.03.2023/binar.tif'
    fn = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/crop/1.tif'
    # 5368930 5367252 5365713 5374896 5364501 5369074 5369449 5362984 5362175 5360740 5379833 5386805
    fn = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/2.tif'
    fn = 'C:/Users/vadik/Desktop/STUDY/diplom/scans/10.03.2023/scan0001.tif'

    img = cv.imread(fn)

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    y_range, x_range, _ = img.shape  # задаем рамзеры картинки
    cv.resizeWindow('result', int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_img, 133, 255, cv.THRESH_BINARY)

    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_lst = []

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат

        area = int(rect[1][0] * rect[1][1])  # вычисление площади

        if 5_550_000 > area > 5_300_000:  # если площадь прямогульника больше  < 9_000_000
        #if 350 > area > 150:
            #print(f'rect params: {rect}')
           # print(area)  # примерно должно быть 6_500_500
            cv.drawContours(img, [box], 0, (0, 0, 255), 2)      # отрисовка прямогуольников размером больше 700_000
            contours_lst.append(area)
            color_yellow = (0, 255, 255)
            center = (int(rect[0][0]), int(rect[0][1]))
            cv.putText(img, f'{area}', (center[0] + 20, center[1] - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 1, color_yellow, 2)

    #cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/rer.tif', img)
    cv.imshow('result', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

    contours_lst.sort()
    print(contours_lst)


from modules import *


def croppp():
    fn = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/scan0007.tif'
    img = cv.imread(fn)

    bin_img = binaryze(img, 160)

    rects = find_contours(bin_img, 2000_000)

    i = 0

    for rect in rects:  # проходимся по найденным контурам

        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат

        x_s = [pair[0] for pair in box]
        y_s = [pair[1] for pair in box]

        x_start, y_start = min(x_s), min(y_s)
        x_stop, y_stop = max(x_s), max(y_s)

        print(x_start, y_start)
        print(x_stop, y_stop)

        crop = img[y_start:y_stop, x_start:x_stop]

        #cv.imwrite(f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/crop/{i}.tif', crop)
        i += 1


def second_crop(path=None, path_to_save=None):
    if path is None:
        path = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/crop/0.tif'
    image = cv.imread(path)
    bin_img_2 = binaryze(image, threshold=130)
    #rects_2 = find_contours(bin_img_2, more_than=292_000, less_then=5_700_000)  # 2000_000
    rects_2 = find_contours(bin_img_2, more_than=5_300_000, less_then=5_550_000)

    rect = rects_2[0]
    box_2 = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box_2 = np.int0(box_2)  # округление координа

    angle = rect[2]

    if angle > 45:  # если неправильный угол изображения
        box_2 = np.insert(box_2, 0, box_2[-1],  axis = 0)     # меняем обход точек
        box_2 = box_2[:-1]

    area_dots = []  # сохряняются количество найденных точек

    for i in range(4):      # по 4 точкам смотрим окрсетности
        a_start_x, a_start_y = box_2[i][0], box_2[i][1]
        angle_crop = image[a_start_y-100:a_start_y+100, a_start_x-100:a_start_x+100]
        area_dots.append(find_circles(angle_crop))

    if analyze_rot(area_dots):   # определяем нужно ли попорачитвать
        crop = crop_image(image, rect, 180)
    else:
        crop = crop_image(image, rect, -90)
    cv.imwrite(path_to_save, crop)
        # #cv.imshow('angcfle', angle_crop)
        # cv.waitKey(0)
        # cv.imwrite(f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/super_crop/{i}.tif', angle_crop)




#croppp()
#contours_finder()


#econd_crop()

# for i in range(10):
#     path = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/crop/{i}.tif'
#     path_to_save = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/rot/{i}.tif'
#     second_crop(path, path_to_save)
#     print()



# for i in range(10):
#     p = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/rot/{i}.tif'
#     r = cv.imread(p)
#     print(r.shape)

def find_circles_analyze():
    def nothing(*arg):
        pass

    fn = f'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/super_crop/3.tif'

    img = cv.imread(fn)

    y_range, x_range, _ = img.shape

    cv.namedWindow("result", cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('result', int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза
    cv.namedWindow("settings")  # создаем окно настроек

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv.createTrackbar('thresh', 'settings', 134, 255, nothing)
    cv.createTrackbar('high', 'settings', 0, 1000, nothing)

    cv.resizeWindow('settings', 700, 140)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    while True:
        img = cv.imread(fn)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


        h1 = cv.getTrackbarPos('thresh', 'settings')
        s1 = cv.getTrackbarPos('high', 'settings')

        ret, thresh = cv.threshold(gray, h1, 255, cv.THRESH_BINARY)

        contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours0:
            perimeter = cv.arcLength(cnt, True)
            if 65 > perimeter > 25:
                ellipse = cv.fitEllipse(cnt)
                print(perimeter)
                cv.ellipse(img, ellipse, (0, 0, 255), 2)

        cv.imwrite('C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/rer.tif', img)

            # cv2.resizeWindow('result', 1000, 1000)  # уменьшаем картинку в 3 раза
        cv.imshow('result', img)

        ch = cv.waitKey(5)
        if ch == 27:
            break

    cv.destroyAllWindows()

#find_circles_analyze()


if __name__ == '__main__':
    #main()

    path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/brightness_96/'
    path_to_save = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/brightness_96/'

    cnt = 10
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    r = [10]
    for i in r:
        p = path + f'scan00{i}.tif'
        scan = cv.imread(p)
        crop, rotated = crop_scans_crop(scan)


        y_range, x_range, _ = crop.shape  # задаем рамзеры картинки
        cv.namedWindow("result crop", cv.WINDOW_NORMAL)  # создаем главное окно
        cv.resizeWindow('result crop', int(x_range // 6), int(y_range // 6))  # уменьшаем картинку в 3 раза

        cv.imshow('result crop', crop)
        #cv.waitKey(0)

        p = path_to_save + f'{cnt}.tif'
        cv.imwrite(p, crop)
        cnt += 1


   #binarization_analyzer()

