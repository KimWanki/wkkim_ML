import os


class Bicycle():

    def __init__(self, wheel_size, color):
        self.wheel_size = wheel_size
        self.color = color

    def move(self, speed):
        print("자전거: 시속 {0}킬로미터로 전진".format(speed))

    def turn(self, direction):
        print("자전거: {0}회전".format(direction))

    def stop(self):
        print("자전거({0}, {1}):정지".format(self.wheel_size, self.color))

class Car2():
    count = 0; # 클래스 변수 생성 및 초기화
    def __init__(self, size, num):
        self.size = size
        self.count = num
        Car2.count = Car2.count + 1 # 클래스 변수 생성 및 초기화
        print("자동차 객체의 수 : Car2.count = {0}".format(Car2.count)) # 클래스 변수 이용
        print("인스턴스 변수 초기화: self.count ={0}".format(self.count))

    def move(self):
        print("자동차({0} & {1})가 움직입니다.".format(self.size, self.count))

my_bicycle = Bicycle(26,'black')

# my_bicycle.wheel_size = 26
# my_bicycle.color = 'black'
#
# print("바퀴 크기:", my_bicycle.wheel_size)
# print("색상:", my_bicycle.color)

my_bicycle.move(30)
my_bicycle.turn('좌')
my_bicycle.stop()

car1 = Car2("big", 20)
car2 = Car2("small", 30)
