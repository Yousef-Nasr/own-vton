import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import uvicorn
import requests
from pyngrok import ngrok
from catvton import EfficientCatVTON  # Make sure this is correct

app = FastAPI()

# Allow all CORS (can be tightened for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your VTON model
virtual_tryon = EfficientCatVTON(device="cuda")

@app.get("/try-on")
async def try_on(
    person_url: str = Query(..., description="URL of the person image"),
    cloth_url: str = Query(..., description="URL of the cloth image"),
    cloth_type: str = Query("upper", description="Type of cloth: upper or lower"),
    num_inference_steps: int = Query(50, description="Number of inference steps"),
):
    try:
        # Download person image
        person_response = requests.get(person_url)
        person_response.raise_for_status()
        person_image = Image.open(io.BytesIO(person_response.content)).convert("RGB")

        # Download cloth image
        cloth_response = requests.get(cloth_url)
        cloth_response.raise_for_status()
        cloth_image = Image.open(io.BytesIO(cloth_response.content)).convert("RGB")

        # Perform virtual try-on
        result_image = virtual_tryon.try_on(
            person_image, cloth_image,
            cloth_type=cloth_type,
            num_inference_steps=num_inference_steps
        )

        # Convert result to bytes
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

