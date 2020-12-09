import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask,render_template,request
from keras.preprocessing import image
from werkzeug.utils import secure_filename


'''config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)'''

tf.compat.v1.keras.backend.get_session
app=Flask(__name__)

@app.route("/",methods=["GET"])
def index():
    return render_template('index.html')



def load_image(img_path):

    img = image.load_img(img_path, target_size=(300, 300))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

@app.route("/upload",methods=["POST"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_name = secure_filename(f.filename)
        path=basepath+'/'+'static/'+file_name
        f.save(path) 
    model = tf.keras.models.load_model("./pred.h5")

    #new_image = load_image('/static/'+file_name)
    new_image = load_image(path)
    # check prediction
    pred = model.predict_classes(new_image)
    class_out='Normal' if pred==[0]  else 'Pneumonia'

    return render_template('index.html',status=class_out,ipath=file_name)

if __name__=='__main__':
    app.run(debug=True)