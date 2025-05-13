import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
import requests
from pyngrok import ngrok
from catvton import EfficientCatVTON  # Make sure this is correct
import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS (can be tightened for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Cloudinary - replace with your actual credentials
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "djgq0eerq"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "673215842768325"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "5NJj8jzUQCinG4IL01V75NT3z9s"),
)

# Initialize your VTON model
virtual_tryon = EfficientCatVTON(device="cuda")

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_url: str = Form(...),
    cloth_type: str = Form("upper"),
    num_inference_steps: int = Form(50),
    folder: Optional[str] = Form("vton_results")
):
    try:
        # Process person image from uploaded file
        person_content = await person_image.read()
        person_img = Image.open(io.BytesIO(person_content)).convert("RGB")
        
        # Download cloth image from URL
        cloth_response = requests.get(cloth_url)
        cloth_response.raise_for_status()
        cloth_img = Image.open(io.BytesIO(cloth_response.content)).convert("RGB")
        
        # Perform virtual try-on
        result_image = virtual_tryon.try_on(
            person_img, cloth_img,
            cloth_type=cloth_type,
            num_inference_steps=num_inference_steps
        )
        
        # Convert result to bytes for Cloudinary upload
        img_io = io.BytesIO()
        result_image.save(img_io, format="PNG")
        img_io.seek(0)
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_io,
            public_id = "try_on_glamora",
            overwrite = True, 
            folder=folder,
            resource_type="image"
        )
        
        # Return the Cloudinary URL and other details
        return {
            "success": True,
            "image_url": upload_result["secure_url"],
            "public_id": upload_result["public_id"],
            "format": upload_result["format"],
            "width": upload_result["width"],
            "height": upload_result["height"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000, domain="seriously-moved-husky.ngrok-free.app")
    print(f"Public URL: {public_url}")
    
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)