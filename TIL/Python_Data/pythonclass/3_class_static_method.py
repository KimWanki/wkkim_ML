class Car():

    instance_count = 0

    def __init__(self, size, color):
        self.size = size
        self.color = color
        Car.instance_count += 1
    def move(self, speed):
        pass
    def auto_cruise(self):
        pass
    @staticmethod
    def check_type(model_code):
        if(model_code >= 20):
            print("이 자동차는 전기차입니다.")
        elif(10 <= model_code < 20):
            print("이 자동차는 가솔린차입니다.")
        else:
            print("이 자동차는 디젤차입니다.")

    @classmethod
    def count_instance(cls):
        print("자동차 객체의 개수 : {0}".format(cls.instance_count))


Car.check_type(25)
Car.count_instance()

car1 = Car("small","black")
Car.count_instance()

car2 = Car("big","white")
Car.count_instance()