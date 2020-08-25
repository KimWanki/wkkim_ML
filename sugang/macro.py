import datetime as dt
import time
import pyautogui
import random
endhope = False

while not endhope:
    tim=dt.datetime.now()
    x,y = pyautogui.position()
    print(x,y)
    pyautogui.click(x,y)

    rand = random.randrange(1, 3)
    time.sleep(rand)