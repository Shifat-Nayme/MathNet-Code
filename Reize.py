import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob
final_word =[]
def function(image):
    global final_word
    image = cv2.resize(image,dsize =(28,28), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #h,w=image.shape
    
    

    final_word.append(image)


images = [cv2.imread(file) for file in glob.glob("C://Users//shifa//Desktop//white data//44//*.jpg")]
print(len(images))

for i in range(len(images)): 
    function(images[i])

outpath ="C://Users//shifa//Desktop//New folder (4)//44//"
idx =1

for j in range(len(final_word)):
    img = final_word[j]
    cv2.imwrite(outpath + str(idx) + '.jpg', img)
    idx = idx + 1