import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob
final_word =[]
def function(image):
    global final_word
    #h,w =image.shape
    #image = cv2.resize(image,dsize =(64,64), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w=image.shape
    
    for i in range (h):
        for j in range (w):
            if(i>=12 and i<= h-11):
                image[i][j]=image[i][j]
            else:
                image[i][j]=0
            if(j>=7 and j<=w-10):
                image[i][j]=image[i][j]
            else:
                image[i][j]=0
            
    #plt.imshow(image)
    #plt.show()
    final_word.append(image)

images = [cv2.imread(file) for file in glob.glob("C://Users//shifa//Desktop//fresh data//45//*.jpg")]
print(len(images))

for i in range(len(images)): 
    function(images[i])

outpath ="C://Users//shifa//Desktop//cut//45//"
idx =0

for j in range(len(final_word)):
    img = final_word[j]
    cv2.imwrite(outpath + str(idx) + '.jpg', img)
    idx = idx + 1