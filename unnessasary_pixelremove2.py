import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob
final_word=[]

def word_segmentation_one(image,size,start,end):
    global final_word
    h,w = image.shape
    img_array = np.zeros((h,size), dtype="uint8")
    x = 0
    y = 0
  
    for i in range(h):
        y = 0
        for j in range(w):
            if(j>=start and j<end):
                img_array[x][y] = image[i][j]
                y = y + 1
            
        x = x + 1
    h2,w2 = img_array.shape
    h,w = img_array.shape
    img = np.stack((img_array,) * 3,-1)
    img = img.astype(np.uint8)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.resize(grayed, dsize =(w,100), interpolation = cv2.INTER_AREA)
    final_word.append(grayed)

def word_segmentation(image):
    h,w=image.shape
    x=10000000
    y=0
    for i in range(h):
        for j in range(w):
            if(image[i][j]>=10):
                if(x>=j):
                    x=j
                    break
            if(image[i][j]>=10):
                if(y<=j):
                    y=j
    word_segmentation_one(image,abs(x-y)+10,x-5,y+5)



def line_segmentation(image,size,letter,start,end):
    h,w = image.shape
    img_array =np.zeros((size,w), dtype="uint8")
    x = 0
    y = 0
    
    for i in range(h):
        if(i>=start and i<end):
            y = 0
            for j in range(w):
                img_array[x][y] = image[i][j]
                y = y + 1 
            x = x + 1
    #plt.imshow(img_array)
    #plt.show()
    return img_array


def function(image):
    #plt.imshow(image)
    #plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array=image
    h,w = image.shape
    cout = 0
    c = 0
    gap = []

    gap.append(0)
    for i in range(h):
        cout = 0
        for j in range(w):
            if(image[i][j] == 0):
                cout = cout + 1
                if cout >= w:
                    c = c + 1
                    gap.append(i)

    gap.append(h)
    letter_size = []
    line_start =[]
    line_end = []
    for i in range(len(gap)-1):
        if(gap[i+1] - gap[i]>20):
            letter_size.append(gap[i+1] - gap[i])
            line_start.append(gap[i])
            line_end.append(gap[i+1])

        
    crop_image = []
    for i in range(len(letter_size)):
        crop = line_segmentation(image,letter_size[i],letter_size,line_start[i],line_end[i])
        h,w = crop.shape
        grayed = cv2.resize(crop, dsize =(1000, 150), interpolation = cv2.INTER_AREA)
        #plt.imshow(crop)
        #plt.show()
        crop_image.append(crop)

    for i in range(len(crop_image)):
        word_segmentation(crop_image[i])


images = [cv2.imread(file) for file in glob.glob("C://Users//shifa//Desktop//cut//18//*.jpg")]
print(len(images))

for i in range(len(images)): 
    function(images[i])

outpath ="C://Users//shifa//Desktop//un//18//"
idx =0

for j in range(len(final_word)):
    img = final_word[j]
    #print(j)
    img = cv2.resize(img,dsize =(28,28), interpolation = cv2.INTER_AREA)
    cv2.imwrite(outpath + str(idx) + '.jpg', img)
    idx = idx + 1

