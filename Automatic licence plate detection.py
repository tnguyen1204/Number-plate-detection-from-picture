#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

print('Library sucessfully imported')

fig,ax = plt.subplots(2,3)

picture_1 = cv2.imread('image1.jpg')
ax[0, 0].imshow(cv2.cvtColor(picture_1, cv2.COLOR_BGR2RGB) if picture_1 is not None else None)
ax[0, 0].set_title('Picture before')
ax[0, 0].axis('off')

picture_2 = cv2.cvtColor(picture_1,cv2.COLOR_BGR2GRAY)
ax[0, 1].imshow(cv2.cvtColor(picture_2, cv2.COLOR_BGR2RGB) if picture_1 is not None else None)
ax[0, 1].set_title('Grayed picture')
ax[0, 1].axis('off')

#Noise reduction
bfilter = cv2.bilateralFilter(picture_2,11,17,17)

#Edge detection 
picture_3 = cv2.Canny(bfilter,30,200)
ax[0, 2].imshow(cv2.cvtColor(picture_3, cv2.COLOR_BGR2RGB) if picture_1 is not None else None)
ax[0, 2].set_title('Edge detection')
ax[0, 2].axis('off')

#Contour detection 
keypoints = cv2.findContours(picture_3,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted (contours,key = cv2.contourArea,reverse=True)[:10] #Recupère les 10 contour qui ont le plus d'air
location = None
for contour in contours :
    approx = cv2.approxPolyDP(contour,5,True)
    #print (len(approx))
    if len(approx) == 4 : #Detection rectangle
        location = approx
        print('Plate number localized !')
        break
    else : 
        print('Plate number NOT localized !')
#print(location)

#Finding the plate 
mask = np.zeros(picture_2.shape, np.uint8) #Created a black picture at the same picture_2
picture_4 = cv2.drawContours(mask,[location],0,255,-1)
picture_4 = cv2.bitwise_and(picture_1, picture_1,mask=mask)
ax[1, 0].imshow(cv2.cvtColor(picture_4, cv2.COLOR_BGR2RGB) if picture_1 is not None else None)
ax[1, 0].set_title('Isolation of the plate')
ax[1, 0].axis('off')

#Cropped the picture only on the plate 
(x,y) = np.where(mask==255) #Finding all the pixel which are not black
(x1,y1) = (np.min(x),np.min(y))
(x2,y2) = (np.max(x),np.max(y))
picture_5 = picture_2[x1:x2+1,y1:y2+1]
ax[1, 1].imshow(cv2.cvtColor(picture_5, cv2.COLOR_BGR2RGB) if picture_1 is not None else None)
ax[1, 1].set_title('Cropped plate')
ax[1, 1].axis('off')

#Read the plate 
plate = easyocr.Reader(['en'])
plate_analyzed = plate.readtext(picture_5)
print(plate_analyzed)
plate_number = plate_analyzed[0][-2]
plate_accuracy = round(plate_analyzed[0][-1],2)*100
print(plate_number)
print(plate_accuracy)

#Print the plate number on the picture
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
text = cv2.putText(picture_1, text = plate_number + ' (accuracy : ' + str(plate_accuracy)+'%)', org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
text = cv2.rectangle(picture_1, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
picture_6 = cv2.cvtColor(text, cv2.COLOR_BGR2RGB)
ax[1, 2].imshow(picture_6)
ax[1, 2].set_title('Plated detected')
ax[1, 2].axis('off')

#cv2.imshow('Automatic plate detection', picture_1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.tight_layout()
plt.show()