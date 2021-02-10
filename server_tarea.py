from flask import Flask
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import cv2, base64

# variables Flask
app = Flask(__name__)
api = Api(app)


# se carga el modelo de Logistic Regression del Notebook #3
pkl_filename = "ModeloLR.pkl"
with open(pkl_filename, 'rb') as file:
    #model = pickle.load(file)
    unpickler = pickle.Unpickler(file)
    model = unpickler.load()

class Predict(Resource):

    @staticmethod
    def get():
        #return 200
        
        parser = reqparse.RequestParser()
        parser.add_argument('image')
        # request para el modelo
        args = parser.parse_args() 
        im_bytes = base64.b64decode(args['image'])
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_GRAYSCALE)
        img = img.reshape(-1)
        # prediccion
        out = {'Prediccion': int(model.predict(img.reshape(1, -1)))}
        return out,200


        
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True, port='1080')