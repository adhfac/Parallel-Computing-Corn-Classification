from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
from multiprocessing.pool import ThreadPool
import numpy as np

MODEL = load_model(
    'model/corn_disease_model.h5',
    compile=False
)

classes = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
]

def predict_single(path):
    img = preprocess_image(path)
    prediction = MODEL.predict(
        img,
        verbose=0
    )
    class_index = np.argmax(
        prediction
    )
    return {
        'image': path,
        'label': classes[
            class_index
        ]
    }

def thread_prediction(
    image_paths,
    workers=4
):
    with ThreadPool(
        processes=workers
    ) as pool:
        results = pool.map(
            predict_single,
            image_paths
        )
    return results