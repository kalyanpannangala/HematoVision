import os
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("Blood_Cell.h5")

class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return class_labels[class_index]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = predict_image(filepath)

            return render_template("result.html",
                                   prediction=result,
                                   image_path=filepath)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)