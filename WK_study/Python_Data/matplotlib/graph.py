import matplotlib.pyplot as plt

# %matplotlib qt        : 그래프를 별도의 팝업창에서 출력
# %matplotlib inline    : 그래프를 코드 결과 출력 부분에 출력

data1 = [10, 14, 19, 20, 25]
# plt.plot(data1)
# plt.show()

import numpy as np

x = np.arange(-4.5, 5, 0.5)  # 배열 x 생성. 범위 : [-4.5, 5) 0.5 간격
y = 2 * x ** 2
print([x, y])

# plt.plot(x,y)
# plt.show()

x = np.arange(-4.5, 5, 0.5)
y1 = 2 * x ** 2
y2 = 5 * x + 30
y3 = 4 * x ** 2 + 10

# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)

# plt.plot(x,y1,x,y2,x,y3)
# plt.show()

# 새로운 그래프 창을 생성해서 그래프를 그리는 예
# plt.plot(x, y1)

# plt.figure()
# plt.plot(x, y2)

# plt.show()

x = np.arange(-5, 5, 0.1)
y1 = x ** 2 - 2
y2 = 20 * np.cos(x) ** 2

# plt.figure(n) : n(정수)를 지정하면 지정된 번호르 그래프 창이 그려지게 된다.
plt.figure(1)
plt.plot(x, y1)

plt.figure(2)
plt.plot(x, y2)
plt.figure(1)
plt.plot(x, y2)

plt.figure(2)
plt.clf()  # 이미 생성된 2번 창에 그려진 모든 그래프를 지움.
plt.plot(x, y1)

plt.show()

#plt.subplot(m,n,p)

