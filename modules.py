import cv2 as cv
import numpy as np


def binaryze(image, threshold=220):
    # y_range, x_range, _ = image.shape
    # cv.namedWindow(f"binaryze {threshold}", cv.WINDOW_NORMAL)  # создаем главное окно
    # cv.resizeWindow(f"binaryze {threshold}", int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза

    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)
    # cv.imshow(f"binaryze {threshold}", thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return thresh


# на вход подается бинаризированное изображение
# на выходе координаты контуров

def find_contours(bin_image, more_than=0, less_then=10_000_000):
    contours0, hierarchy = cv.findContours(bin_image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        area = int(rect[1][0] * rect[1][1])  # вычисление площади прямоугольника

        if less_then > area > more_than:
            rectangles.append(rect)

            box = cv.boxPoints(rect)
            box = np.int0(box)  # округление координат
            cv.drawContours(bin_image, [box], 0, (0, 0, 255), 8)
            y_range, x_range = bin_image.shape
            cv.namedWindow("bin_image", cv.WINDOW_NORMAL)  # создаем главное окно
            cv.resizeWindow('bin_image', int(x_range // 8), int(y_range // 8))  # уменьшаем картинку в 3 раза
            cv.imshow('bin_image', bin_image)
            cv.waitKey(0)
            print(area)

    #if len(rectangles) >= 2:
    #    del rectangles[0]  # удаляем максимальный прямогуольник
    # (если правильно понял, то первым в списке идет прямоугольник с максимальной площадью)
    # если нет, то надо будет подумать над обработкой
    return rectangles


def crop_image(img, rect, add_angle=0):
    # на вход подать изображение и область (распознанную) по которой обрезать
    # rect ->  ((центр прямогуольника), (размер прямоугольника), угол наклона)
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    if add_angle < 0:  # если не надо переворачивать
        size = (size[1], size[0])
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    angle += add_angle
    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)  # формирует матрицу поворота
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))  # вращает изображение

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop


def find_circles(img):
    bin_image = binaryze(img, 134)
    contours0, hierarchy = cv.findContours(bin_image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    res = []

    for cnt in contours0:
        perimeter = cv.arcLength(cnt, True)
        if 68 > perimeter > 30:
            ellipse = cv.fitEllipse(cnt)
            res.append(ellipse)
        #print(perimeter)
    return len(res)


def count_squares(img):
    bin_image = binaryze(img, 133)
    cnt = find_contours(bin_image, more_than=140, less_then=380)
    return len(cnt)


def analyze_rot(lst):
    if sum(lst[1:3]) > lst[0] + lst[-1]:
        #print('need to rotate')
        return True
    elif lst[0] + lst[-1] > sum(lst[1:3]):
        #print('no need')
        return False
    else:
        #print('error')
        return False


def crop_scans(img):
    bin_img = binaryze(img, 155)
    rects = find_contours(bin_img, 2000_000)
    cropped_imgs = []

    for rect in rects:  # проходимся по найденным контурам

        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат

        x_s = [pair[0] for pair in box]
        y_s = [pair[1] for pair in box]

        x_start, y_start = min(x_s), min(y_s)
        x_stop, y_stop = max(x_s), max(y_s)

        crop = img[y_start:y_stop, x_start:x_stop]

        cropped_imgs.append(crop)

    return cropped_imgs


def crop_scans_crop(image):
    #bin_img = binaryze(image, threshold=141)
    bin_img = binaryze(image, threshold=145)
    rects = find_contours(bin_img, more_than=5_300_000, less_then=5_550_000)
    #rects = find_contours(bin_img, more_than=5766000, less_then=6449536)

    rect = rects[0]
    box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.int0(box)  # округление координа
    angle = rect[2]

    if angle > 45:  # если неправильный угол изображения
        box = np.insert(box, 0, box[-1],  axis=0)     # меняем обход точек
        box = box[:-1]

    area_dots = []  # сохряняются количество найденных точек

    for i in range(4):      # по 4 точкам смотрим окрсетности
        a_start_x, a_start_y = box[i][0], box[i][1]
        angle_crop = image[a_start_y-100:a_start_y+100, a_start_x-100:a_start_x+100]

        cv.imshow('angle_crop', angle_crop)

        area_dots.append(find_circles(angle_crop))

    rotated = analyze_rot(area_dots)

    if rotated:   # определяем нужно ли попорачитвать
        crop = crop_image(image, rect, 180)
    else:
        crop = crop_image(image, rect, -90)

    return crop, rotated  # возвращает изображение и флаг повернуто оно или нет


def main():
    show_crop_res = False
    path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/'
    path = 'C:/Users/vadik/Desktop/STUDY/diplom/scans/10.03.2023/'

    path = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/brightness_32/'
    path_to_save = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/ultra/'
    path_to_save = 'C:/Users/Root/Documents/MEGAsync/diplom/scans/10.03.2023/brightness_32/'

    cnt = 0
    r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in r:
        p = path + f'scan000{i}.tif'
        scan = cv.imread(p)
        cropped_scan = crop_scans(scan)

        if len(cropped_scan) > 0:
            print(f'Find {len(cropped_scan)} images on:  {p[p.rfind("/") + 1:]}')
        else:
            print(f'Images to crop on:  {p[p.rfind("/") + 1:]} not find!')
            continue

        for cropped in cropped_scan:
            res, is_rot = crop_scans_crop(cropped)
            p = path_to_save + f'{cnt}.tif'

            if is_rot:
                print(f'Saved pic: {p} | rotated')
                cv.imwrite(p, res)
            else:
                print(f'Saved pic {p} | original')
                cv.imwrite(p, res)

            cnt += 1
            if show_crop_res:
                y_range, x_range, _ = res.shape  # задаем рамзеры картинки
                cv.namedWindow("result crop", cv.WINDOW_NORMAL)  # создаем главное окно
                cv.resizeWindow('result crop', int(x_range // 6), int(y_range // 6))  # уменьшаем картинку в 3 раза

                cv.imshow('result crop', res)
                cv.waitKey(0)

        print('-' * 40)
        print()

#main()

def show_contours(bin_image, more_than=0, less_then=10000_000_000):
    contours0, hierarchy = cv.findContours(bin_image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   # contours0, hierarchy = cv.findContours(bin_image.copy(),  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    edged = cv.Canny(bin_image, 15, 50)
    #contours0, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.imshow('Canny Edges After Contouring', edged)
    contours0, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        area = int(rect[1][0] * rect[1][1])  # вычисление площади прямоугольника
        print(area)

        if less_then > area > more_than:
            rectangles.append(rect)

            box = cv.boxPoints(rect)
            box = np.int0(box)  # округление координат
            cv.drawContours(bin_image, [box], 0, (0, 0, 255), 2)
            y_range, x_range = bin_image.shape
            cv.namedWindow("bin_image", cv.WINDOW_NORMAL)  # создаем главное окно
            cv.resizeWindow('bin_image', int(x_range // 8), int(y_range // 8))  # уменьшаем картинку в 3 раза


    cv.imshow('bin_image', bin_image)
    cv.waitKey(0)
            # print(area)


# p = 'C:/Users/vadik/Desktop/STUDY/diplom/scans/10.03.2023/2.png'
# #p = 'C:/Users/vadik/Desktop/STUDY/diplom/scans/10.03.2023/brightness_0/scan0001.tif'
# img = cv.imread(p)
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# show_contours(gray_img)


# 1) исходное изображение бинаризуется
# 2) найденные контуры вырезаются
# 3) проходимся по каждому найденному контуру
# 4) снова бинаризуем изображение
# 5) находим минимальный контур, по кот. далее будет обрезаться конечное изображение
# 6) проходимся по углам минимального найденного контура и смотрим их окрестности на наличие реперных знаков
# 7) определяем по расположению реперных знаков поворот и определеяем добавочный угол
