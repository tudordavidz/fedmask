import random
import string
import cv2
from Useful_Tools import *
from PIL import Image


def getNameFile(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    #print("Random alphanumeric String is:", result_str)
    return result_str

def SaveNoCropImagesFromCamera(getFrames):
    path =  pathToProcess('/dataCollection/')
    photoName = getNameFile(3)
    cv2.imwrite(path + 'NoCrop/{index}.png'.format(index=photoName),getFrames)
    name = (path + 'NoCrop/{index}.png'.format(index=photoName))
    image = Image.open(name)
    return image
        
def SaveWithMask(image, tupleCoordinates):
    region = image.crop(tupleCoordinates)
    path =  pathToProcess('/dataCollection/')
    photoName = getNameFile(3)
    region.save(path + "WithMask/{index}.png".format(index=photoName))
    #print("SAVE MASK")
    return photoName
    
        
def SaveNoMask(image, tupleCoordinates):
    region = image.crop(tupleCoordinates)
    path =  pathToProcess('/dataCollection/')
    photoName = getNameFile(3)
    region.save(path + "NoMask/{index}.png".format(index=photoName))
    #print("SAVE NO MASK")
    return photoName
            
def SaveFederatedDataColection(labelMask, getFrames, tupleCoordinates):
    image = SaveNoCropImagesFromCamera(getFrames)
    photoName = "Bianca"
    Mask = False
    NoMask = False
    if labelMask == 'Mask':
        photoName = SaveWithMask(image, tupleCoordinates)
        Mask = True
    if labelMask == 'NoMask':
        photoName = SaveNoMask(image, tupleCoordinates)
        NoMask = True
    return (photoName, Mask, NoMask)
    
    

    
            
            