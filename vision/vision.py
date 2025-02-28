import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

def preprocess_image(image):
    # Implement your image preprocessing steps here
    # For example, resizing, normalization, etc.
    # This is a placeholder implementation
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_object(image):
    # Load the pre-trained model
    model = load_model('model.h5')
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Classify the image
    prediction = model.predict(image)
    
    # Return the predicted class
    return np.argmax(prediction)

def get_class_label(class_index):
    # Load the ImageNet class index file
    with open('imagenet_class_index.json') as f:
        class_index_dict = json.load(f)
    
    # Get the class label
    class_id = str(class_index)
    class_label = class_index_dict[class_id][1]
    return class_label

if __name__ == "__main__":
    # Example usage
    image_path = 'image.jpg'
    image = Image.open(image_path)
    predicted_class = detect_object(image)
    class_label = get_class_label(predicted_class)
    print(f'Predicted class: {predicted_class} ({class_label})')