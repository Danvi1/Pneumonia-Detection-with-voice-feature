from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pyttsx3

app = Flask(__name__)

# Load the trained model
model = load_model('model\our_vgg16_model.h5')

@app.route('/')
def index():
    return render_template('index.html')



# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the uploaded image file
#     file = request.files['image']

#     # Load and preprocess the image
#     img = image.load_img(file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_data = preprocess_input(img_array)

#     # Make the prediction
#     prediction = model.predict(img_data)
#     if prediction[0][0] > prediction[0][1]:
#         result = 'Person is safe.'
#     else:
#         result = 'Person is affected with Pneumonia.'

#     return render_template('index.html', result=result, prediction=prediction[0])

def voice(result):
    engine = pyttsx3.init()
    engine.say(result)
    engine.runAndWait()
    engine.stop()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        # Save the uploaded file to a temporary location
        file.save('temp_image.jpg')
        
        # Load the image from the temporary file
        img = image.load_img('temp_image.jpg', target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_data = preprocess_input(img_array)

        prediction = model.predict(img_data)
        result = 'Person is safe.' if prediction[0][0] > prediction[0][1] else 'Person is affected with Pneumonia.'
        voice(result)

        return render_template('index.html', result=result, prediction=prediction[0])
    else:
        return render_template('index.html', error='Error processing the file')


if __name__ == '__main__':
    app.run(debug=True)