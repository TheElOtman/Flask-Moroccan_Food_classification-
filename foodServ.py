from keras.models import model_from_json
import flask
import werkzeug
import cv2
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = None


app = flask.Flask(__name__)
def get_predict(image):
	global model
	if model == None:
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights("model.h5")
	X_test = []
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	m, l,_ = image.shape
	if l > m :
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	image = cv2.resize(image,(int(120),int(200)))/255
	X_test.append(image)
	X_test = np.array(X_test)
	disease_cls = ['couscous', 'hercha', 'tajin', 'atay']
	res = disease_cls[model.predict(X_test)[0].argmax()]
	return str(res)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
	imagefile = flask.request.files['image']
	filename = werkzeug.utils.secure_filename(imagefile.filename)
	print("\nReceived image File name : " + imagefile.filename)
	imagefile.save(filename)
	image = cv2.imread(filename)
	return get_predict(image)

def main():
	print("this is mai functions")
	
app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == '__main__':
	main()