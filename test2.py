from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def load_image(img_path):

    img = image.load_img(img_path, target_size=(300, 300))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor
    
def show(img_path,name):
    image=cv2.imread(img_path)
    image = cv2.resize(image,(300,300))
    label_position = (3,299)
    lbl_color=(0,255,0) if name=='Normal' else (0,0,255) 
    cv2.putText(image,name,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,lbl_color,2)
    cv2.imshow('prediction',image)
    cv2.waitKey(9000)
    cv2.destroyAllWindows()  


if __name__ == "__main__":

    # load model
    model = load_model("./pred.h5")

    # image path
    img_path = './chest_xray/val/NORMAL/NORMAL2-IM-1437-0001.jpg'    
    

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict_classes(new_image)
    class_out='Normal' if pred==[0]  else 'Pneumonia'
    print('pred=',class_out)
    show(img_path,class_out)