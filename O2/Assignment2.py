import numpy as np
from PIL import Image
## Task 4: Prediction

f1 = open("/Program_Files/frame25.png","rb")
f2 = Image.open("~/Program_Files/frame26.png")
f3 = Image.open("/Program_Files/frame27.png")

if f1:
    print("Hei")
else:
    print("Nei")

