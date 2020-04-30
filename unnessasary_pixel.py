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
    #plt.imshow(grayed)
    #plt.show()
    final_word.append(grayed)



def word_segmentation(image):
    word_gap = []
    c = 0
    cout = 0
    sum = []
    null_list = []
    h,w = image.shape
    copy = image
    sum.append(copy.sum(axis = 0))
    a = np.asarray(sum)
    m,n = a.shape
    #print(m,n)
    gapp = []
    for i in range(m):
        for j in range(n-1):
            if(a[i][j]<50):
                null_list.append(a[i][j])
                gapp.append(j)
    
    gapp.append(w)
    #print(gapp)
    word_size = []
    word_start =[]
    word_end = []
    for i in range(len(gapp)-1):
        if(gapp[i+1] - gapp[i]>35):
            word_size.append(gapp[i+1] - gapp[i])
            word_start.append(gapp[i])
            word_end.append(gapp[i+1])
    print(word_start)
    for i in range(len(word_size)):
        word_segmentation_one(image,word_size[i],word_start[i],word_end[i])

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
        #grayed = cv2.resize(crop, dsize =(1000, 150), interpolation = cv2.INTER_AREA)
        #plt.imshow(crop)
        #plt.show()
        crop_image.append(crop)

    for i in range(len(crop_image)):
        word_segmentation(crop_image[i])



images = [cv2.imread(file) for file in glob.glob("C://Users//shifa//Desktop//cut//33//*.jpg")]
print(len(images))

for i in range(len(images)): 
    function(images[i])


outpath ="C://Users//shifa//Desktop//un//50//"
idx =0

for j in range(len(final_word)):
    img = final_word[j]
    img = cv2.resize(img,dsize =(28,28), interpolation = cv2.INTER_AREA)
    cv2.imwrite(outpath + str(idx) + '.jpg', img)
    idx = idx + 1