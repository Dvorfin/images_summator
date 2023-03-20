import numpy as np
import matplotlib.pyplot as plt







with open('coords.txt', 'r') as file:
    data = file.readlines()

x = np.array([])
y = np.array([])
for item in data:
    item = item.strip()
    #item = int(item[:item.find('|')])
    x_t = int(item[:item.find('|')])
    y_t = int(item[item.find('|')+2:])

    x = np.append(x, x_t)
    y = np.append(y, y_t)



plt.grid(True)
plt.scatter(x, y, s=2, c='green')  # игрик инвертирован из за матплотлиба
# вычисляем коэффициенты

N = len(x)

mx = x.sum() / N
my = y.sum() / N
a2 = np.dot(x.T, x) / N
a11 = np.dot(x.T, y) / N

kk = (a11 - mx * my) / (a2 - mx ** 2)
bb = my - kk * mx

print(kk, bb)

ff = np.array([kk*z+bb for z in range(len(x))])
x = np.int0(x)\

print(x)
print(ff)
plt.plot(x, ff, c='blue')

plt.show()