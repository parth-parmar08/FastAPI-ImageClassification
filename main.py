from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import uvicorn
import os

#from typing import Annotated

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the pre-trained model
model = load_model("digitSignLanguage(1).h5")

# Define image size expected by the model
#img_size = (128, 128)

@app.get("/", response_class=HTMLResponse)
async def home(request: HTMLResponse):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Process the image
    image = await file.read()
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img = img.resize((128, 128), resample=Image.BICUBIC)
    img_array = np.array(img) / 255.0
    #img_array = np.expand_dims(img_array, axis=0)
    img_reshape = img_array.reshape((1, 128, 128, 3))

    #  Display the reshaped image (optional, if you want to visualize it)
    #Image.fromarray((img_reshape[0] * 255).astype(np.uint8)).show()

    # Make prediction
    prediction = model.predict(img_reshape)
    predicted_class = np.argmax(prediction)

    # Return the predicted class as response
    return {"predicted_class": int(predicted_class)}

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
