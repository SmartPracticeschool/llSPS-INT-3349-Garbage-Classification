import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("garbage.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',secure_filename(f.filename))
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        model._make_predict_function() 
        
        with graph.as_default():
            preds = model.predict_classes(x)
           
            
            
            print("prediction",preds)
            
        index = ['cardboard','glass','metal','paper','plastic','trash']
        
        text = "the predicted waste category is : " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
        
        
        
    
    
    
