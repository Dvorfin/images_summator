import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math


# функция вычисления k и b коэф-тов прямой
def calc_mnk_k_b(x, y):
    x = np.int0(x)  # преобразование икса в инт
    x = np.array([[x] for x in x])  # преобразование икса в столбец

    col = np.array([[1] for _ in range(len(x))])  # создание второго столбца
    x = np.append(x, col, axis=1)
    xTx = np.matmul(x.T, x)  # произведение матриц (xT * x)
    inv_x = np.linalg.inv(xTx)  # взятие обратной матрицы
    tetta = np.matmul(np.dot(inv_x, x.T), y)  # (xT * x)^T * xT * y

   # print(f'k = {tetta[0]}\nb = {tetta[1]}')

    return tetta[0], tetta[1]

names = ['line_0', 'line_1', 'line_2', 'line_3']


# функция вычисления среднего угла и коэф-тов (k, b)
def calc_avg_angle(lines_data):
    avg_angle = 0
    lines_coef = []     # списко коэффициентов прямых (k, b)

    for i in range(4):  # проходимся по 4 линиям
        x = np.array([])
        y = np.array([])

        for item in lines_data[i]:

            x_t = int(item[0])
            y_t = int(item[1])

            x = np.append(x, y_t)
            y = np.append(y, x_t)

        k, b = calc_mnk_k_b(x, y)   # вычисления коэф-ов k и b
        lines_coef.append([k, b])

        if k > 0:   # если острый или тупой угол
            angle = math.atan(k) + (math.pi / 2)
            angle = math.degrees(angle)
         #   print(f'rot angle: {angle}')

            # пересчет в углы в коордианатах opencv
        else:
            angle = math.pi - math.atan(-1 * k)
            angle = math.degrees(angle)
          #  print(f'rot angle: {angle}')

        if angle < 135:     # пересчет в углы в коордианатах opencv
            avg_angle += (90 - angle)
        else:
            avg_angle += (180 - angle)

        #print()

    print(f'res angle = {avg_angle / 4}')

    return avg_angle / 4, lines_coef


# функция вычисляет точки пересечения 4 линий для наъождения 4 точек углов прямоугольника
# на вход подаются коэф-ты (k и b)
def calc_intersection_points(lines):
    rect_points = []

    for i in range(4):  # проходимся по 4 линиям и ищим точки их пересечения
        k1, b1 = lines[i - 1][0], lines[i - 1][1]
        k2, b2 = lines[i][0], lines[i][1]

        y = (b2 - b1) / (k1 - k2)
        x = k1 * ((b2 - b1) / (k1 - k2)) + b1
       # print(f'x = {x}')
       # print(f'y = {y}')
        rect_points.append([x, y])
        print(f'{i + 1} angle point: {x} | {y}')

        #print()
    return rect_points


# функция вычисляет центр прямоугольника
# на вход подаются коэф-ты (k и b)
def calc_center(rect_points):
    x1, y1 = rect_points[0]
    x2, y2 = rect_points[2]

    k1 = (y2 - y1) / (x2 - x1)
    b1 = (y1 + y2 - k1 * (x1 + x2)) / 2

    x3, y3 = rect_points[1]
    x4, y4 = rect_points[3]

    k2 = (y4 - y3) / (x4 - x3)
    b2 = (y3 + y4 - k2 * (x3 + x4)) / 2

    x = (b2 - b1) / (k1 - k2)
    y = k1 * ((b2 - b1) / (k1 - k2)) + b1

    return x, y


def calc_rect_size(rect_points):
    x1, y1 = rect_points[0]
    x2, y2 = rect_points[2]
    x3, y3 = rect_points[1]
    x4, y4 = rect_points[3]

    # вычисляем ширину верхней и нижней линии и берем среднее
    width_1 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
    width_2 = ((x2 - x4) ** 2 + (y2 - y4) ** 2) ** 0.5
    width = (width_1 + width_2) / 2
    print(f'Width: {width}')

    # аналогично для высоты
    height_1 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    height_2 = ((x1 - x4) ** 2 + (y1 - y4) ** 2) ** 0.5
    height = (height_1 + height_2) / 2
    print(f'Height: {height}')

    return width, height


def test():

    lines_coef = []

    data = []
    for i in range(4):
        with open(f'line{i}.txt', 'r') as file:
            data.append(file.readlines())


    avg_angle = 0

    for i in range(4):
        x = np.array([])
        y = np.array([])

        for item in data[i]:
            item = item.strip()
            #item = int(item[:item.find('|')])
            x_t = int(item[:item.find('|')])
            y_t = int(item[item.find('|')+2:])


            # x = np.append(x, x_t)
            # if i % 2:
            #     y = np.append(y, -y_t)
            # else:
            #     y = np.append(y, -y_t)
            x = np.append(x, y_t)
            y = np.append(y, x_t)

        plt.grid(True)
        #plt.scatter(x, y, s=5, c='green')  # игрик инвертирован из за матплотлиба

        print(names[i])
        k, b = calc_mnk_k_b(x, y)
        f = np.array([k*x + b for x in x])
        #plt.plot(x, f, c='blue')
        lines_coef.append([k, b])

        #print(f'angle: {math.atan(k)}')
        if k > 0:
            #print(f'0 angle: {math.degrees(math.atan(k))}')
            angle = math.atan(k) + (math.pi / 2)
            angle = math.degrees(angle)
            print(f'rot angle: {angle}')

            #print(f'angle radians: {angle}')
            #print(f'angle degrees: {180 - math.degrees(angle)}')
            #print(f'angle degrees: {180 - angle}')
        else:
           # print(f'0 angle: {math.degrees(math.atan(-1 * k))}')
            angle = math.pi - math.atan(-1 * k)
            angle = math.degrees(angle)
            print(f'rot angle: {angle}')


            # if math.degrees(angle) > 90:
            #     angle = 90 - angle
            #print(f'angle radians: {angle}')
            #print(f'angle degrees: {180 - math.degrees(angle)}')
            #print(f'angle degrees: {180 - angle}')

        if angle < 135:
            avg_angle += (90 - angle)
        else:
            avg_angle += (180 - angle)

            #print(f'angle: {(-math.pi / 2) + math.atan(k) }')
            #print(f'angle: {(-math.pi / 2) + math.pi - math.atan(1 * k)}')

        print()

        path = 'Screenshot_3.png'
        img = cv.imread(path)
        y_range, x_range, _ = img.shape
        cv.namedWindow('countours', cv.WINDOW_NORMAL)  # создаем главное окно
        cv.resizeWindow('countours', int(x_range // 1), int(y_range // 1))  # уменьшаем картинку в 3 раза


        for y, x in zip(x, f):
            cv.circle(img, (int(x), int(y)), 1, (0, 0, 255), 10)
        #cv.line(img, (int(x[5]), int(f[5])), (int(x[500]), int(f[500])), (0,0,255), 5)

        cv.imshow('countours', img)
        #cv.waitKey(0)
        cv.destroyAllWindows()
        #plt.show()

    print(f'res angle = {avg_angle / 4}')

    return lines_coef

if __name__ == '__main__':
    # print(calc_avg_angle())
    lines = test()

    print(lines)

    # x1 = [i for i in range(500)]
    # y1 = [lines[0][0] * x + lines[0][1] for x in x1]
    #
    # plt.plot(x1, y1)
    #
    # y2 = [i for i in range(500)]
    # x2 = [lines[1][0] * x + lines[1][1] for x in y2]
    #
    # plt.plot(x2, y2)

    path = 'Screenshot_3.png'
    #path = '2.tif'
    img = cv.imread(path)
    y_range, x_range, _ = img.shape
    cv.namedWindow('countours', cv.WINDOW_NORMAL)  # создаем главное окно
    cv.resizeWindow('countours', int(x_range // 7), int(y_range // 7))  # уменьшаем картинку в 3 раза

    rect_points = []

    for i in range(4):  # проходимся по 4 линиям и ищим точки их пересечения
        k1, b1 = lines[i - 1][0], lines[i - 1][1]
        k2, b2 = lines[i][0], lines[i][1]

        y = (b2 - b1) / (k1 - k2)
        x = k1 * ((b2 - b1) / (k1 - k2)) + b1
        print(f'x = {x}')
        print(f'y = {y}')

        rect_points.append([x, y])

        cv.circle(img, (int(x), int(y)), 3, (0, 0, 255), 12)
        print()


    print(rect_points)

    x1, y1 = rect_points[0]
    x2, y2 = rect_points[2]

    k1 = (y2 - y1) / (x2 - x1)
    b1 = (y1 + y2 - k1*(x1 + x2)) / 2

    x3, y3 = rect_points[1]
    x4, y4 = rect_points[3]

    k2 = (y4 - y3) / (x4 - x3)
    b2 = (y3 + y4 - k2 * (x3 + x4)) / 2

    plt.scatter(x1, y1)
    plt.text(x1 + 2, y1 + 2, 'A')

    plt.scatter(x2, y2)
    plt.text(x2 + 2, y2 + 2, 'B')

    plt.scatter(x3, y3)
    plt.text(x3 + 2, y3 + 2, 'C')

    plt.scatter(x4, y4)
    plt.text(x4 + 2, y4 + 2, 'D')




    print(f'k1, b1 = {k1} {b1}')
    print(f'k2, b2 = {k2} {b2}')

    x = (b2 - b1) / (k1 - k2)
    y = k1 * ((b2 - b1) / (k1 - k2)) + b1

    print(f'center: {x}, {y}')

    plt.scatter(x, y)
    plt.text(x + 2, y + 2, 'O')  # из за других координат в matplotlib x и y местами поменяны

    cv.circle(img, (int(x), int(y)), 5, (0, 255, 255), 30)


    cv.imshow('countours', img)
    cv.imwrite('res_points.tif', img)


    width = ((x1-x3)**2 + (y1-y3)**2)**0.5
    print(f'width: {width}')

    height = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    print(f'height: {height}')

    #print(lines)

    rect = ((x, y), (width, height), 16.42407013943044)

    plt.show()

    cv.waitKey(0)

    # на вход подать изображение и область (распознанную) по которой обрезать
    def crop_rot_rect(img, rect):
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

    crop = crop_rot_rect(img, rect)

    cv.imshow('countours', crop)
    cv.imwrite('crop_res.tif', crop)
    #cv.waitKey(0)