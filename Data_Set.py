import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import glob

final_word = []
cnt = 0
c=0
li =[]

def word_segmentation_two(image,size,letter,start,end):
    global final_word
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

    h,w = img_array.shape
    img = np.stack((img_array,) * 3,-1)
    img = img.astype(np.uint8)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayed = cv2.resize(grayed, dsize =(w,100), interpolation = cv2.INTER_AREA)
    li.append(grayed)


def word_segmentation_one(image,size,start,end):
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

    img_array_rry = np.array(img_array)

    for i in range(h2):
        for j in range(w2):
            if(img_array[i][j]>=1):
                img_array[i][j] = 255
            else:
                img_array[i][j] = 0   


    cout = 0
    c = 0
    gap = []

    gap.append(0)
    for i in range(h2):
        cout = 0
        for j in range(w2):
            if(img_array[i][j] == 0):
                cout = cout +1
                if cout >= w:
                    c = c + 1
                    gap.append(i)

    gap.append(h2)
    word_size = []
    word_start =[]
    word_end = []
    total_gap = 0
    for i in range(len(gap)-1):
        if(gap[i+1] - gap[i]== 1):
            total_gap = total_gap + 1
        if(gap[i+1] - gap[i]>25):
            word_size.append(gap[i+1] - gap[i])
            word_start.append(gap[i])
            word_end.append(gap[i+1])
   
    for i in range(len(word_size)):
        word_segmentation_two(img_array_rry,word_size[i],size,word_start[i],word_end[i])


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
            if(a[i][j]<10):
                null_list.append(a[i][j])
                gapp.append(j)
    
    gapp.append(w)
    word_size = []
    word_start =[]
    word_end = []
    for i in range(len(gapp)-1):
        if(gapp[i+1] - gapp[i]>25):
            word_size.append(gapp[i+1] - gapp[i])
            word_start.append(gapp[i])
            word_end.append(gapp[i+1])
    
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
    return img_array



from PIL import Image

def function(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    img_array = np.zeros((h,w), dtype="uint8")

    for i in range(h):
        for j in range(w):
            if gray[i][j]>1:
                img_array[i][j] = gray[i][j]
            else:
                img_array[i][j] = 0
    ret, thresh = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 

    print("\n Resizing Image........")
    global cnt
    cnt=cnt+1
    print(cnt)
    image = cv2.resize(thresh, dsize =(2500,3500), interpolation = cv2.INTER_AREA)
    #plt.imshow(image)
    #plt.show()

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
        if(gap[i+1] - gap[i]>5):
            letter_size.append(gap[i+1] - gap[i])
            line_start.append(gap[i])
            line_end.append(gap[i+1])

        
    crop_image = []
    for i in range(len(letter_size)):
        crop = line_segmentation(image,letter_size[i],letter_size,line_start[i],line_end[i])
        h,w = crop.shape
        #grayed = cv2.resize(crop, dsize =(1000, 150), interpolation = cv2.INTER_AREA)
        crop_image.append(crop)

    for i in range(len(crop_image)):
        word_segmentation(crop_image[i])


images = [cv2.imread(file) for file in glob.glob("C://Users//Public//Documents//ScanDoc//6//*.jpg")]
print(len(images))

for i in range(len(images)): 
    function(images[i])
    print(len(li))
    if(len(li)==90):
        for i in range(90):
            final_word.append(li[i])
    li.clear()

outpath ="C://Users//shifa//Desktop//fresh data//1//"
outpath1 ="C://Users//shifa//Desktop//fresh data//2//"
outpath2 ="C://Users//shifa//Desktop//fresh data//3//"
outpath3 ="C://Users//shifa//Desktop//fresh data//4//"
outpath4 ="C://Users//shifa//Desktop//fresh data//5//"
outpath5 ="C://Users//shifa//Desktop//fresh data//6//"
outpath6 ="C://Users//shifa//Desktop//fresh data//7//"
outpath7 ="C://Users//shifa//Desktop//fresh data//8//"
outpath8 ="C://Users//shifa//Desktop//fresh data//9//"
outpath9 ="C://Users//shifa//Desktop//fresh data//10//"
outpath10 ="C://Users//shifa//Desktop//fresh data//11//"
outpath11 ="C://Users//shifa//Desktop//fresh data//12//"
outpath12 ="C://Users//shifa//Desktop//fresh data//13//"
outpath13 ="C://Users//shifa//Desktop//fresh data//14//"
outpath14 ="C://Users//shifa//Desktop//fresh data//15//"
outpath15 ="C://Users//shifa//Desktop//fresh data//16//"
outpath16 ="C://Users//shifa//Desktop//fresh data//17//"
outpath17 ="C://Users//shifa//Desktop//fresh data//18//"
outpath18 ="C://Users//shifa//Desktop//fresh data//19//"
outpath19 ="C://Users//shifa//Desktop//fresh data//20//"
outpath20 ="C://Users//shifa//Desktop//fresh data//21//"
outpath21 ="C://Users//shifa//Desktop//fresh data//22//"
outpath22 ="C://Users//shifa//Desktop//fresh data//23//"
outpath23 ="C://Users//shifa//Desktop//fresh data//24//"
outpath24 ="C://Users//shifa//Desktop//fresh data//25//"
outpath25 ="C://Users//shifa//Desktop//fresh data//26//"
outpath26 ="C://Users//shifa//Desktop//fresh data//27//"
outpath27 ="C://Users//shifa//Desktop//fresh data//28//"
outpath28 ="C://Users//shifa//Desktop//fresh data//29//"
outpath29 ="C://Users//shifa//Desktop//fresh data//30//"
outpath30 ="C://Users//shifa//Desktop//fresh data//31//"
outpath31 ="C://Users//shifa//Desktop//fresh data//32//"
outpath32 ="C://Users//shifa//Desktop//fresh data//33//"
outpath33 ="C://Users//shifa//Desktop//fresh data//34//"
outpath34 ="C://Users//shifa//Desktop//fresh data//35//"
outpath35 ="C://Users//shifa//Desktop//fresh data//36//"
outpath36 ="C://Users//shifa//Desktop//fresh data//37//"
outpath37 ="C://Users//shifa//Desktop//fresh data//38//"
outpath38 ="C://Users//shifa//Desktop//fresh data//39//"
outpath39 ="C://Users//shifa//Desktop//fresh data//40//"
outpath40 ="C://Users//shifa//Desktop//fresh data//41//"
outpath41 ="C://Users//shifa//Desktop//fresh data//42//"
outpath42 ="C://Users//shifa//Desktop//fresh data//43//"
outpath43 ="C://Users//shifa//Desktop//fresh data//44//"
outpath44 ="C://Users//shifa//Desktop//fresh data//45//"

idx = 24000
i = 0
model_image = []
img_width = []
print(len(final_word))
for j in range(len(final_word)):
    img = final_word[j]
    #plt.imshow(img)
    #plt.show()
    if(i==1):
         cv2.imwrite(outpath + str(idx) + '.jpg', img)
    if(i==3):
        cv2.imwrite(outpath1 + str(idx) + '.jpg', img)
    if(i==5):
        cv2.imwrite(outpath2 + str(idx) + '.jpg', img)
    if(i==7):
        cv2.imwrite(outpath3 + str(idx) + '.jpg', img)
    if(i==9):
        cv2.imwrite(outpath4 + str(idx) + '.jpg', img)
    if(i==11):
        cv2.imwrite(outpath5 + str(idx) + '.jpg', img)
    if(i==13):
        cv2.imwrite(outpath6 + str(idx) + '.jpg', img)
    if(i==15):
        cv2.imwrite(outpath7 + str(idx) + '.jpg', img)
    if(i==17):
        cv2.imwrite(outpath8 + str(idx) + '.jpg', img)
    if(i==19):
        cv2.imwrite(outpath9 + str(idx) + '.jpg', img)
    if(i==21):
        cv2.imwrite(outpath10 + str(idx) + '.jpg', img)
    if(i==23):
        cv2.imwrite(outpath11 + str(idx) + '.jpg', img)
    if(i==25):
        cv2.imwrite(outpath12 + str(idx) + '.jpg', img)
    if(i==27):
        cv2.imwrite(outpath13 + str(idx) + '.jpg', img)
    if(i==29):
        cv2.imwrite(outpath14 + str(idx) + '.jpg', img)
    if(i==31):
        cv2.imwrite(outpath15 + str(idx) + '.jpg', img)
    if(i==33):
        cv2.imwrite(outpath16 + str(idx) + '.jpg', img)
    if(i==35):
        cv2.imwrite(outpath17 + str(idx) + '.jpg', img)
    if(i==37):
        cv2.imwrite(outpath18 + str(idx) + '.jpg', img)
    if(i==39):
        cv2.imwrite(outpath19 + str(idx) + '.jpg', img)
    if(i==41):
        cv2.imwrite(outpath20 + str(idx) + '.jpg', img)
    if(i==43):
        cv2.imwrite(outpath21 + str(idx) + '.jpg', img)
    if(i==45):
        cv2.imwrite(outpath22 + str(idx) + '.jpg', img)
    if(i==47):
        cv2.imwrite(outpath23 + str(idx) + '.jpg', img)
    if(i==49):
        cv2.imwrite(outpath24 + str(idx) + '.jpg', img)
    if(i==51):
        cv2.imwrite(outpath25 + str(idx) + '.jpg', img)
    if(i==53):
        cv2.imwrite(outpath26 + str(idx) + '.jpg', img)
    if(i==55):
        cv2.imwrite(outpath27 + str(idx) + '.jpg', img)
    if(i==57):
        cv2.imwrite(outpath28 + str(idx) + '.jpg', img)
    if(i==59):
        cv2.imwrite(outpath29 + str(idx) + '.jpg', img)
    if(i==61):
        cv2.imwrite(outpath30 + str(idx) + '.jpg', img)

    if(i==63):
        cv2.imwrite(outpath31 + str(idx) + '.jpg', img)
    if(i==65):
        cv2.imwrite(outpath32 + str(idx) + '.jpg', img)
    if(i==67):
        cv2.imwrite(outpath33 + str(idx) + '.jpg', img)
    if(i==69):
        cv2.imwrite(outpath34 + str(idx) + '.jpg', img)
    if(i==71):
        cv2.imwrite(outpath35 + str(idx) + '.jpg', img)
    if(i==73):
        cv2.imwrite(outpath36 + str(idx) + '.jpg', img)
    if(i==75):
        cv2.imwrite(outpath37 + str(idx) + '.jpg', img)
    if(i==77):
        cv2.imwrite(outpath38 + str(idx) + '.jpg', img)
    if(i==79):
        cv2.imwrite(outpath39 + str(idx) + '.jpg', img)
    if(i==81):
        cv2.imwrite(outpath40 + str(idx) + '.jpg', img)
    if(i==83):
        cv2.imwrite(outpath41 + str(idx) + '.jpg', img)
    if(i==85):
        cv2.imwrite(outpath42 + str(idx) + '.jpg', img)
    if(i==87):
        cv2.imwrite(outpath43 + str(idx) + '.jpg', img)
    if(i==89):
        cv2.imwrite(outpath44 + str(idx) + '.jpg', img)
    i = i+1
    if(i==90):
       i=0
    idx = idx + 1

print("complete")