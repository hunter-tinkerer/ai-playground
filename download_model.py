from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import save_model

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Save the model to a file
model.save('model.h5')