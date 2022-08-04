
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.lib.utils import source
import math
import tensorflow as tf

from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from keras.models import model_from_json
# from keras.models import load_model

# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)

    if(math.fabs(angle)>60):

        angle+=math.degrees(math.pi)/2
    
    return rotateImage(cvImage, -1.0 * angle)


  
# Function to generate horizontal projection profile
def getHorizontalProjectionProfile(image):
  
    # Convert black spots to ones
    image[image == 0]   = 1
    # Convert white spots to zeros
    image[image == 255] = 0
  
    horizontal_projection = np.sum(image, axis = 1) 
  
    return horizontal_projection
  
def getVerticalProjectionProfile(image):
  
    # Convert black spots to ones 
    image[image == 0]   = 1
    # Convert white spots to zeros 
    image[image == 255] = 0
  
    vertical_projection = np.sum(image, axis = 0)
  
    return vertical_projection
# Driver Function

def shadow_remove(img):
      rgb_planes = cv2.split(img)
      result_norm_planes = []
      for plane in rgb_planes:
          dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
          bg_img = cv2.medianBlur(dilated_img, 21)
          diff_img = 255 - cv2.absdiff(plane, bg_img)
          norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
          result_norm_planes.append(norm_img)
      shadowremov = cv2.merge(result_norm_planes)
      return shadowremov


def prediction2(image):
   
    
    # characters = '०,१,२,३,४,५,६,७,८,९,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ'
    # characters = characters.split(',')
    
    labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','क्ष','त्र','ज्ञ',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']
    
   
    # plt.imshow(image,cmap='gray')
    # plt.show()
    
    
    image = cv2.resize(image,(32,32))
    image=cv2.GaussianBlur(image,(3,3),0)
    ret,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image[0:2,:]=255
    plt.imshow(image,cmap='gray')
    plt.show()
    
    image = img_to_array(image)
   
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    model =     tf.keras.models.load_model("charModel.h5") 
    lists = model.predict(image/255.0)[0]
    # print(np.argmax(lists))
    # print("The letter is ",labels[np.argmax(lists)])    
    return labels[np.argmax(lists)],lists[(np.argmax(lists))]*100,np.argmax(lists)

def prediction(img):
  
    loaded_model=load_model('cnn2.hdf5')
    
    characters = '०,१,२,३,४,५,६,७,८,९,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ'
    characters = characters.split(',')
    image= cv2.resize(img,(32,32))
    image=cv2.GaussianBlur(image,(3,3),0)
    ret,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image[0:3,:]=255
   

  

    image = img_to_array(image)
   
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    
    
   
    
    output = loaded_model.predict(image/255.0)
    output = output.reshape(46)
    predicted = np.argmax(output)
    devanagari_label = characters[predicted]
    success = output[predicted] * 100
    # print(predicted)
    
    return devanagari_label, success,predicted


def predict(image):    
    image = shadow_remove(image)       
    image=deskew(image)
 
    image = shadow_remove(image)
    image=cv2.resize(image,(600,200))
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
 
    ret,image=cv2.threshold(image,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
    kernel=np.ones((1,1),np.uint8)    
    image=cv2.dilate(image,kernel,iterations=1)   
    image1=image.copy()   
    horizontal_projection = getHorizontalProjectionProfile(image1)   
    a=0
    b=len(horizontal_projection)
    for i in range(a,b-1):
        if(horizontal_projection[i]==0 and horizontal_projection[i+1]!=0):
            a=i
            break;
    for i in range(0,b-1):
        if(horizontal_projection[b-i-1]==0 and horizontal_projection[b-i-2]!=0):
            b=b-i-1
            break
    image=image[a:b,:]    
    image1=image.copy()
    slice=int((b-a)*0.25)
    image1=image1[slice:,:]
    image2=image1.copy()     
    vertical_projection= getVerticalProjectionProfile(image1)
    c=0
    d=len(vertical_projection)
    for i in range(c,d-1):
        if(vertical_projection[i]==0 and vertical_projection[i+1]!=0):
            c=i
            break;
    for i in range(0,d-1):
        if(vertical_projection[d-i-1]==0 and vertical_projection[d-i-2]!=0):
            d=d-i-1
            break    
    image=image[:,c:d]
    cv2.imshow("img",image)
    cv2.waitKey(0)
 
    vertical_projection= getVerticalProjectionProfile(image2[:,c:d])   
    found0=[]
    zero_start=0
    zero_end=0
    for i in range(0,len(vertical_projection)):        
        if(vertical_projection[i]==0 and zero_start==0):
            zero_start=i            
        if(vertical_projection[i]!=0 and zero_start!=0):
            zero_end=i
            found0.append(int(zero_start+(zero_end-zero_start)/2))
            zero_start=0

 
    found0.append(len(vertical_projection))
    found0.insert(0,0)
    print(found0)

    image3=image.copy()
    image_segmented=[]
    for i in range(len(found0)-1):
        image_segmented.append(image[:,found0[i]:found0[i+1]])
        cv2.line(image3,(found0[i],0),(found0[i],20),(0,0,0),1)
    

    cv2.imshow("o",image3)
    cv2.waitKey(0)
    string=""
    return_list=[]
    print(image_segmented)
    for i in image_segmented:
        
        im=cv2.bitwise_not(i)
        
        # kernel=np.ones((1,1),np.uint8)    
        # image=cv2.dilate(image,kernel,iterations=1)   
        
        im2=im.copy()    
        var1,var2,num=prediction2(im)
        var3,var4,num2=prediction(im2)
        print(var1,var3)
        if num in[10,29]:
            string+=' '+str(var1)
            return_list.append(var1)

        elif num2 in [15,44]:
            string+=' '+str(var3)
            return_list.append(var3)

        elif var2>var4:
            string+=' '+str(var1)
            return_list.append(var1)
        else :
            string+=' '+str(var3)
            return_list.append(var3)
            

       
    cv2.destroyAllWindows()

    return return_list
def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = ['./images/1.jpeg']


    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a list is expected
        print(' '.join(answer))# will be the output string

      
    
  
   
if __name__ == '__main__':
    test()
    
   


    

