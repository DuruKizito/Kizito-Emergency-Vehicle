from flask import Flask, request, render_template_string
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = 'kme93.keras'
model = load_model(model_path)

def predict_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))
    label = "Emergency" if pred > 0.5 else "Non-Emergency"
    return label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            prediction = predict_image(image)
            return render_template_string("""
                <html>
                <body>
                    <h1>Welcome to Kizito's Emergency Vehicle Detection Server</h1>
                    <p>Prediction: {{ prediction }}</p>
                    <form action="/" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept="image/*">
                        <input type="submit" value="Open Image">
                    </form>
                </body>
                </html>
            """, prediction=prediction)
    return render_template_string("""
        <html>
        <body>
            <h1>Welcome to Kizito's Emergency Vehicle Detection Server</h1>
            <form action="/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Open Image">
            </form>
        </body>
        </html>
    """)

#if __name__ == '__main__':
#    app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
