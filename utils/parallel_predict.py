from multiprocessing import Pool
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
import numpy as np

classes = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
]

def predict_single(path):

    model = load_model('model/corn_disease_model.h5')

    img = preprocess_image(path)

    prediction = model.predict(img, verbose=0)

    class_index = np.argmax(prediction)

    return {
        'image': path,
        'label': classes[class_index]
    }

def parallel_prediction(image_paths):

    model = load_model(
        'model/corn_disease_model.h5'
    )

    images=[]

    for path in image_paths:

        img = preprocess_image(path)[0]

        images.append(img)

    batch=np.array(images)

    predictions=model.predict(
        batch,
        verbose=0
    )

    results=[]

    for i,pred in enumerate(predictions):

        class_index=np.argmax(pred)

        results.append({
            'image': image_paths[i],
            'label': classes[class_index]
        })

    return results