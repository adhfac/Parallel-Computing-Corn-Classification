from multiprocessing import Pool, cpu_count
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
import numpy as np

classes = [
    'Blight',
    'Common_Rust',
    'Gray_Leaf_Spot',
    'Healthy'
]

MODEL = None

def init_worker():
    global MODEL
    MODEL = load_model(
        'model/corn_disease_model.h5'
    )


def predict_chunk(image_paths):

    global MODEL
    images=[]
    for path in image_paths:
        img=preprocess_image(path)[0]
        images.append(img)
    batch=np.array(images)
    predictions=MODEL.predict(
        batch,
        verbose=0
    )
    results=[]
    for i,pred in enumerate(predictions):
        class_index=np.argmax(pred)
        results.append({
            'image':image_paths[i],
            'label':classes[class_index]
        })
    return results


def mp_prediction(
    image_paths,
    workers=4,
    chunk_size=16
):
    chunks=[
        image_paths[i:i+chunk_size]
        for i in range(
            0,
            len(image_paths),
            chunk_size
        )
    ]
    with Pool(
        processes=workers,
        initializer=init_worker
    ) as pool:
        results=pool.map(
            predict_chunk,
            chunks
        )
    flattened=[]
    for batch_result in results:
        flattened.extend(
            batch_result
        )
    return flattened