from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

import numpy as np
from multiprocessing.pool import ThreadPool

model = load_model('model/corn_disease_model.h5')

classes = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
]

def predict_single(path):

    img = preprocess_image(path)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)

    label = classes[class_index]

    return {
        'image': path,
        'label': label
    }

def parallel_prediction(image_paths, batch_size=4):

    pool = ThreadPool(processes=batch_size)

    results = pool.map(predict_single, image_paths)

    pool.close()
    pool.join()

    return results