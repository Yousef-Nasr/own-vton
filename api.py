import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from pyngrok import ngrok
from catvton import EfficientCatVTON  # Ensure this module exists

from fastapi.responses import StreamingResponse

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

virtual_tryon = EfficientCatVTON(device="cuda")

@app.post("/try-on")
async def try_on(person: UploadFile = File(...), cloth: UploadFile = File(...), cloth_type: str = "upper"):
    try:
        # Read uploaded images
        person_image = Image.open(io.BytesIO(await person.read())).convert("RGB")
        cloth_image = Image.open(io.BytesIO(await cloth.read())).convert("RGB")
        
        # Perform virtual try-on
        result_image = virtual_tryon.try_on(person_image, cloth_image, cloth_type=cloth_type, num_inference_steps=50)
        
        # Convert result image to bytes
        img_io = io.BytesIO()
        result_image.save(img_io, format="PNG")
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000, domain="seriously-moved-husky.ngrok-free.app")
    print(f"Public URL: {public_url}")

    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
