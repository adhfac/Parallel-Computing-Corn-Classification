from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
import numpy as np

model = load_model(
    'model/corn_disease_model.h5'
)

classes = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
]

def serial_prediction(image_paths):

    results=[]

    for path in image_paths:

        img = preprocess_image(path)

        prediction = model.predict(
            img,
            verbose=0
        )

        class_index=np.argmax(prediction)

        label=classes[class_index]

        results.append({
            'image':path,
            'label':label
        })

    return results