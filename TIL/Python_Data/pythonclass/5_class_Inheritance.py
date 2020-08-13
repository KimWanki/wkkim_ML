from WK_study.Python_Data.pythonclass.class_def import Bicycle

class FoldingBicycle(Bicycle):

    def __init__(self, wheel_size, color, state):
        Bicycle.__init__(self, wheel_size, color)
        #super().__init__(wheel_size, color) # super()도 사용 가능
        self.state = state #자식 클래스에서 새로 추가한 변수
        #단, super()을 사용하는 경우, 인자에서 self를 빼야함.

    def fold(self):
        self.state = 'folding'
        print("자전거: 접기, state = {0}".format(self.state))

    def unfold(self):
        self.state = 'unfolding'
        print("자전거: 접기, state = {0}".format(self.state))

folding_bicycle = FoldingBicycle(27, 'black', 'unfolding')

folding_bicycle.move(20)
folding_bicycle.fold()
folding_bicycle.unfold()

