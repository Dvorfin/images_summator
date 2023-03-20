import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#fffc cv from ito_lr_1 import *

def binaryze(image, threshold=220):
    y_range, x_range, _ = image.shape
    cv.namedWindow(f"binaryze {threshold}", cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow(f"binaryze {threshold}", int(x_range // 2), int(y_range // 2))  # уменьшаем картинку в 3 раза

    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)
    cv.imshow(f"binaryze {threshold}", thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thresh


def redefine_rect_points(box):
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # Four point coordinates up, down, left, and right
    vertices = np.array([[left_point_x, left_point_y], [top_point_x, top_point_y], [right_point_x, right_point_y],
                         [bottom_point_x, bottom_point_y]])

    return vertices


def find_contours(img, more_than=0, less_then=2000):
    bin = binaryze(img, 133)
    contours0, hierarchy = cv.findContours(image=bin.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    for cnt in contours0:
        if less_then > cnt.shape[0] > more_than:    # если размер контура найден
            #cv.drawContours(img, cnt, -1, (0, 255, 0), 2, cv.LINE_AA) # отрсиовка найденного контура
            rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
            box = cv.boxPoints(rect)
            box = np.int0(box)  # округление координат
            #print(box)

            print(cnt.shape[0])

            A, B, C, D = box[0], box[1], box[2], box[3]


            x_s = [point[0][0] for point in cnt]
            y_s = [point[0][1] for point in cnt]

            arr = np.empty((0, 2), int)
            for p in cnt:
                for point in p:
                    arr = np.append(arr, [[point[0], point[1]]], axis=0)
            #print(arr)
            #C = min(x_s), min(y_s)
            #B = max(x_s), max(y_s)

            max_element_column = np.max(arr, 0)
            max_element_row = np.max(arr, 1)

            min_element_column = np.amin(arr, 0)
            min_element_row = np.amin(arr, 1)

            # printing the result
            print('max x, y:', max_element_column)
            print('min x, y:', min_element_column)


            print()

            x = 1000000
            y = 1000000
            for p in cnt:
                for point in p:
                    x_p, y_p = point
                    x_min, y_min = min_element_column

                    if x_p == x_min:
                        if y_p < y:
                            y = y_p

                    if y_p == y_min:
                        if x_p < x:
                            x = x_p

            A1 = [min_element_column[0], y]
            B1 = [x, min_element_column[1]]
            print(A1, B1)

            x = 0
            y = 100000
            for p in cnt:
                for point in p:
                    x_p, y_p = point
                    x_max, y_max = max_element_column

                    if x_p == x_max:
                        #print(f' y = {y_p}')
                        if y_p < y:
                            y = y_p

                    if y_p == y_max:
                        #print(f' x = {x_p}')
                        if x_p > x:
                            x = x_p

                        #print(f' x = {x_p}')

            C1 = [max_element_column[0], y]
            D1 = [x, max_element_column[1]]
            print(C1, D1)



            # x_r_min, x_r_max = min(A1[0], B1[0]), max(A1[0], B1[0])
            # y_r_min, y_r_max = min(A1[1], B1[1]), max(A1[1], B1[1])

            # x_r_min, x_r_max = min(B1[0], C1[0]), max(B1[0], C1[0])
            # y_r_min, y_r_max = min(B1[1], C1[1]), max(B1[1], C1[1])

            # x_r_min, x_r_max = min(C1[0], D1[0]), max(C1[0], D1[0])
            # y_r_min, y_r_max = min(C1[1], D1[1]), max(C1[1], D1[1])

            # x_r_min, x_r_max = min(A1[0], D1[0]), max(A1[0], D1[0])
            # y_r_min, y_r_max = min(A1[1], D1[1]), max(A1[1], D1[1])






            # x_r_min, x_r_max = min(A[0], B[0]), max(A[0], B[0])
            # y_r_min, y_r_max = min(A[1], B[1]), max(A[1], B[1])

            # x_r_min, x_r_max = min(B[0], C[0]), max(B[0], C[0])
            # y_r_min, y_r_max = min(B[1], C[1]), max(B[1], C[1])

            # x_r_min, x_r_max = min(C[0], D[0]), max(C[0], D[0])
            # y_r_min, y_r_max = min(C[1], D[1]), max(C[1], D[1])

            # x_r_min, x_r_max = min(D[0], A[0]), max(D[0], A[0])
            # y_r_min, y_r_max = min(D[1], A[1]), max(D[1], A[1])

            # print(f'x range: {x_r_min}, {x_r_max}')
            # print(f'y range: {y_r_min}, {y_r_max}')


            colors = [(10, 0, 204), (150, 10, 0), (0, 255, 0), (255, 255, 0)]


            angles = [[A1, B1], [B1, C1], [C1, D1], [D1, A1]]

            for i in range(4):
                point1, point2 = angles[i]
                x_r_min, x_r_max = min(point1[0], point2[0]), max(point1[0], point2[0])
                y_r_min, y_r_max = min(point1[1], point2[1]), max(point1[1], point2[1])


                for p in cnt:
                    with open('text.txt', 'a+', encoding='utf-8') as test:
                        for point in p:
                            point = np.int0(point)
                            # test.write(str(r) + '\n')
                            if (x_r_min+1 < point[0] < x_r_max -1) and (y_r_min+1 < point[1] < y_r_max-1):
                                #print(point)
                                #cv.drawContours(img, cnt, -1, (0, 255, 0), 2, cv.LINE_AA)   # отрсиоввываем найденный контур
                                cv.circle(img, point, 1, colors[i], 2)


                #cv.drawContours(img, [box], 0, (255, 0, 100), 2)
                for point in box:
                        #cv.circle(img, point, 5, (250, 200, 100), 2)  # рисуем маленький кружок в центре прямоугольника
                        # выводим в кадр величину угла наклона
                        cv.putText(img, f'{point}', (point[0] - 40, point[1] + 20),
                                                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)

    y_range, x_range, _ = img.shape
    cv.namedWindow('countours', cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('countours', int(x_range // 9), int(y_range // 9))  # уменьшаем картинку в 3 раза
    cv.imshow('countours', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


    #cv.imshow('countours', img)


def find_contours_rebuild(img, more_than=0, less_then=2000):
    bin = binaryze(img, 133)
    contours0, hierarchy = cv.findContours(image=bin.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    # массив, куда будут записаны координаты 4-х линий прямоугольника
    lines_of_rectrangle = [[], [], [], []]

    for cnt in contours0:
        if less_then > cnt.shape[0] > more_than:    # если размер контура найден

            print(f'Size of contour: {cnt.shape[0]}')

            arr = np.empty((0, 2), int) # пересобираем массив точек контура в формат [[x1, y1], [x2, y2], [x3, y3]....]
            for p in cnt:
                for point in p:
                    arr = np.append(arr, [[point[0], point[1]]], axis=0)

            #  находим верхнюю левую (нижнюю левую) и нижнюю правую точки (верхнюю правую)
            # (в зависимости от наклона прямогуоьника)
            max_element_column = np.max(arr, 0)
            min_element_column = np.amin(arr, 0)

            # printing the result
            print('max x, y:', max_element_column)
            print('min x, y:', min_element_column)

            print()

            # вычилсляем координаты ближайшей точки к верхней левой (нижней левой)
            x = 0   # доплить надо отвчеает за смещение
            y = 1000000
            for p in cnt:
                for point in p:
                    x_p, y_p = point
                    x_min, y_min = min_element_column

                    if x_p == x_min:
                        if y_p < y:
                            y = y_p

                    if y_p == y_min:
                        if x_p > x:
                            x = x_p

            A1 = [min_element_column[0], y]
            B1 = [x, min_element_column[1]]

            # вычилсляем координаты ближайшей точки к нижней правой (верхней правой)
            x = 0  # ноль, т.к. в правой части обе точки должны иметь икс ближе к правой части
            y = 100000
            for p in cnt:
                for point in p:
                    x_p, y_p = point
                    x_max, y_max = max_element_column

                    if x_p == x_max:
                        #print(f' y = {y_p}')
                        if y_p < y:
                            y = y_p

                    if y_p == y_max:
                        #print(f' x = {x_p}')
                        if x_p > x:
                            x = x_p



            C1 = [max_element_column[0], y]
            D1 = [x, max_element_column[1]]

            print(f'A coord: {A1}')
            print(f'B coord: {B1}')
            print(f'C coord: {C1}')
            print(f'D coord: {D1}')


            # цвета линий
            colors = [(10, 0, 204), (150, 10, 0), (0, 255, 0), (255, 255, 0)]

            # координаты угов
            angles = [[A1, B1], [B1, C1], [C1, D1], [D1, A1]]

            # проходимся по всем углам
            for i in range(4):
                point1, point2 = angles[i]  # берем 2 соседние точки
                # вычисялем ширину и высоту прямогуольника, построенного по 2 угловым точкам
                x_r_min, x_r_max = min(point1[0], point2[0]), max(point1[0], point2[0])
                y_r_min, y_r_max = min(point1[1], point2[1]), max(point1[1], point2[1])

                # проходимся по точкам контура
                for p in cnt:
                    for point in p:
                        point = np.int0(point)
                        # если точка контура лежит в пределах прямогольника, то записываем ее
                        if (x_r_min+1 < point[0] < x_r_max-1) and (y_r_min+1 < point[1] < y_r_max-1):
                            #print(point)
                            #cv.drawContours(img, cnt, -1, (0, 255, 0), 2, cv.LINE_AA)   # отрсиоввываем найденный контур
                            cv.circle(img, point, 1, colors[i], 2)
                            lines_of_rectrangle[i].append(point)


                #cv.drawContours(img, [box], 0, (255, 0, 100), 2)



    y_range, x_range, _ = img.shape
    cv.namedWindow('countours', cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('countours', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза
    cv.imshow('countours', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return lines_of_rectrangle


def find_countours_rebuild_2(img, more_than=0, less_then=2000):
    bin = binaryze(img, 133)
    contours0, hierarchy = cv.findContours(image=bin.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

    # массив, куда будут записаны координаты 4-х линий прямоугольника
    lines_of_rectrangle = [[], [], [], []]
    # множество с точками контура
    coords_set = set()

    for cnt in contours0:
        if less_then > cnt.shape[0] > more_than:  # если размер контура найден
            print(f'Size of contour: {cnt.shape[0]}')
            for p in cnt:
                for point in p:
                    coords_set.add((point[0], point[1]))

    # поиск точки с минимальным иксом
    A = sorted(coords_set)
    min_x = A[0][0]  # выбираем минимальный x
    A1 = []
    for p in A:  # проходимся по всем точкам с минимальными иксами
        if p[0] != min_x:  # если минимальный x изменился
            break
        A1.append(p[1])     # добавляем значения y
    A = (min_x, sum(A1) // len(A1))     # записываем минимальный x и средний y

    # поиск точки с минимальным игриком
    B = sorted(coords_set, key=lambda point: point[1])
    #print(B)
    min_y = B[0][1]
    B1 = []
    for p in B:
        if p[1] != min_y:
            break
        B1.append(p[0])
    B = (sum(B1) // len(B1), min_y)

    # поиск точки с максимальным иксом
    C = sorted(coords_set, reverse=True)
    max_x = C[0][0]
    C1 = []
    for p in C:
        if p[0] != max_x:
            break
        C1.append(p[1])
    C = (max_x, sum(C1) // len(C1))

    # поиск точки с максимальным игриком
    D = sorted(coords_set, reverse=True, key=lambda point: point[1])
    max_y = D[0][1]
    D1 = []
    for p in D:
        if p[1] != max_y:
            break
        D1.append(p[0])
    D = (sum(D1) // len(D1), max_y)

    print(f'Min x: {A}')
    print(f'Min y: {B}')
    print(f'Max x: {C}')
    print(f'Max y: {D}')

    # отрисовка 4 точек с их координатами
    angle_points = [A, B, C, D]
    for point in angle_points:
        cv.circle(img, point, 5, (0, 250, 0), 2)
        cv.putText(img, f'{point}', (point[0] - 40, point[1] + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)


#--------------------------------------------------------------
    for cnt in contours0:
        if less_then > cnt.shape[0] > more_than:
            # цвета линий
            colors = [(10, 0, 204), (150, 10, 0), (0, 255, 0), (255, 255, 0)]

            # координаты угов
            angles = [[A, B], [B, C], [C, D], [D, A]]

            # проходимся по всем углам
            for i in range(4):
                point1, point2 = angles[i]  # берем 2 соседние точки
                # вычисялем ширину и высоту прямогуольника, построенного по 2 угловым точкам
                x_r_min, x_r_max = min(point1[0], point2[0]), max(point1[0], point2[0])
                y_r_min, y_r_max = min(point1[1], point2[1]), max(point1[1], point2[1])

                # проходимся по точкам контура
                for p in cnt:
                    for point in p:
                        point = np.int0(point)
                        # если точка контура лежит в пределах прямогольника, то записываем ее
                        if (x_r_min + 1 < point[0] < x_r_max - 1) and (y_r_min + 1 < point[1] < y_r_max - 1):
                            # print(point)
                            # cv.drawContours(img, cnt, -1, (0, 255, 0), 2, cv.LINE_AA)   # отрсиоввываем найденный контур
                            cv.circle(img, point, 1, colors[i], 2)
                            lines_of_rectrangle[i].append(point)

                # cv.drawContours(img, [box], 0, (255, 0, 100), 2)

    y_range, x_range, _ = img.shape
    cv.namedWindow('countours', cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('countours', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза
    cv.imshow('countours', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('res.tif', img)
    return lines_of_rectrangle

   # cv.imshow('res', img)

def coefficient_reg_inv(x, y):
    size = len(x)
    # формируем и заполняем матрицу размерностью 2x2
    A = np.empty((2, 2))
    A[[0], [0]] = sum((x[i]) ** 2 for i in range(0, size))
    A[[0], [1]] = sum(x)
    A[[1], [0]] = sum(x)
    A[[1], [1]] = size
    # находим обратную матрицу
    A = np.linalg.inv(A)
    # формируем и заполняем матрицу размерностью 2x1
    C = np.empty((2, 1))
    C[0] = sum((x[i] * y[i]) for i in range(0, size))
    C[1] = sum((y[i]) for i in range(0, size))

    # умножаем матрицу на вектор
    ww = np.dot(A, C)
    return ww[1], ww[0]


if __name__ == '__main__':

    # path = '1.png'
    # img = cv.imread(path)
    # #lines = find_contours_rebuild(img, 1600, 2000)
    # find_angles(img, 1600, 2000)

    # path = '1.png'
    # img = cv.imread(path)
    # find_angles(img, 1600, 2000)

    # path = '2.png'
    # img = cv.imread(path)
    # find_countours_rebuild_2(img, 1700, 2000)


    #path = '2.tif'
    #path = '0.tif'

    # path = 'scan1.png'
    # img = cv.imread(path)
    # #lines = find_contours_rebuild(img, 2000, 4000)
    # find_countours_rebuild_2(img, 2000, 4000)

    # path = 'scan2.png'
    # img = cv.imread(path)
    # # lines = find_contours_rebuild(img, 3600, 3700)
    # find_countours_rebuild_2(img, 3600, 3700)


    # path = '0.tif'
    # img = cv.imread(path)
    #find_countours_rebuild_2(img, 9000, 12300)


    path = '2.tif'
    img = cv.imread(path)
    find_countours_rebuild_2(img, 9000, 9791)

    # Size
    # of
    # contour: 12308
    # Size
    # of
    # contour: 9587



    #find_contours(img, 1000, 1130)
    #find_contours(img, 9000, 9600)

    #lines = find_contours_rebuild(img, 9000, 9600)


   # item = np.int0(lines[0])
    # with open('coords.txt', 'w', encoding='utf-8') as file:
    #     for line in item:
    #         x, y = line
    #         srr = f'{x} | {y}\n'
    #         file.write(srr)
   # print(item)


    # x = [x[0] for x in lines[0]]
    # print(x)

    # y = np.array([x[1] for x in lines[0]])
    # np.array([1, 1, 1])

    # print(coefficient_reg_inv(x, y))
    # k, b = coefficient_reg_inv(x, y)


    # def predict(k, b, x_scale):
    #     y_pred = [k * val + b for val in x_scale]
    #     return y_pred
    #
    # y_predict = predict(k, b, x)
    #
    # plt.plot(x, y, 'o', label='Истинные значения')
    # plt.plot(x, y_predict, '*', label='Расчетные значения')
    # plt.legend(loc='best', fontsize=12)
    # plt.xlabel('x (порядковый номер измерения)', fontsize=14)
    # plt.ylabel('y (оценка температуры)', fontsize=14)
    # plt.show()



 # ----------------------------------------------------------------------


    # fig, ax = plt.subplots()
    # points = [[130, 631], [146, 123], [992, 148], [977, 656]]
    # points = [[point[0], 1000 - point[1]] for point in points]
    #
    # x_points = [point[0] for point in points]
    # y_points = [point[1] for point in points]
    #
    # plt.plot(x_points, y_points, 'ro')
    #
    # print(points)
    # A, B, C, D = points[0], points[1], points[2], points[3]
    # print(A, B)
    # plt.plot([A[0], B[0]], [A[1], B[1]], color='blue')
    #
    #
    # plt.axis([0, 1000, 0, 1000])
    # plt.show()

#----------------------------------------------------------------------



    # bin = binaryze(img, 130)
    # # more_than = 1800
    # # less_then = 3000
    # more_than = 1300
    # less_then = 2000
    #
    #
    # contours0, hierarchy = cv.findContours(image=bin.copy(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    #
    #
    #
    # for cnt in contours0:
    #
    #     if cnt.shape[0] > more_than and cnt.shape[0] < less_then: # проверяем размер контура
    #         print(cnt.shape[0])
    #     if less_then > cnt.shape[0] > more_than:    # если размер контура найден
    #         cv.drawContours(img, cnt, -1, (0, 255, 0), 2, cv.LINE_AA)   # отрсиоввываем найденный контур
    #         print(cnt)
    #
    # cv.imshow('bin_image', img)



    # for cnt in contours0:
    #     rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
    #     area = int(rect[1][0] * rect[1][1])  # вычисление площади прямоугольника
    #
    #     print(area)
    #
    #     if less_then > area > more_than:
    #
    #         box = cv.boxPoints(rect)
    #         box = np.int0(box)  # округление координат
    #         cv.drawContours(img, [box], 0, (0, 255, 100), 2)
    #
    #
    #         cnt = 1
    #         for point in box:
    #             cv.circle(img, point, 5, (250, 200, 100), 2)  # рисуем маленький кружок в центре прямоугольника
    #             # выводим в кадр величину угла наклона
    #             cv.putText(img, str(cnt), point,
    #                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
    #             cv.putText(img, f'{point}', (point[0] - 40, point[1] + 20),
    #                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)
    #             cnt += 1
    #
    #
    #
    #
    #         y_range, x_range = bin.shape
    #         cv.namedWindow("bin_image", cv.WINDOW_NORMAL)  # создаем главное окно
    #         cv.resizeWindow('bin_image', int(x_range // 1), int(y_range // 1))  # уменьшаем картинку в 3 раза
    #         cv.imshow('bin_image', img)


    cv.waitKey(0)