from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('mnist_cnn_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'})

    # Read the image
    img = Image.open(io.BytesIO(file.read())).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    

    # Predict the digit
    prediction = model.predict(img)
    digit = int(np.argmax(prediction[0]))

    return jsonify({'prediction': digit})

if __name__ == '__main__':
    app.run(debug=True)
