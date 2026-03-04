import numpy as np

def detect(model, sample):

    probabilities = model.predict_proba(sample)
    prediction = model.predict(sample)

    confidence = np.max(probabilities)

    return prediction[0], confidence