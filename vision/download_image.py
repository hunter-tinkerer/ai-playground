import requests
from PIL import Image
from io import BytesIO

# URL of a sample car image
url = "https://imageio.forbes.com/specials-images/imageserve/5d35eacaf1176b0008974b54/2020-Chevrolet-Corvette-Stingray/0x0.jpg?format=jpg&crop=4560,2565,x790,y784,safe&width=1440"

# Download the image
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Save the image
image.save('image.jpg')