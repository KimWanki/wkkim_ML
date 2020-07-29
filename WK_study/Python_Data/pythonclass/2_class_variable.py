class Car():
    instance_count = 0

    def __init__(self, size, color):
        self.size = size
        self.color = color
        Car.instance_count = Car.instance_count+1
        print("자동차 객체의 수: {0}".format(Car.instance_count))

    def move(self, speed):
        print("자동차({0}, {1})가 움직입니다.".format(self.size, self.color))

car1 = Car('small', 'white')
car2 = Car('big', 'black')

