import io
import logging
import os
from typing import Optional
import asyncio
import json
import random
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
from PIL import Image
import configparser

# Import necessary functions from imageGen
from imageGen import (
    ImageGenerator, 
    search_for_nodes_with_key, 
    edit_given_nodes_properties, 
    get_host, 
    upload_image,
    save_images
)

# Import credit and payment functions
from payment_service import deduct_credits, discord_balance_prompt, discord_recharge_prompt
from db import init_db, get_user_images

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the database
init_db()

# Load configuration
config = configparser.ConfigParser()
config.read("config.properties")

text2img_config = config["LOCAL_TEXT2IMG"]["CONFIG"]

class TextToImageRequest(BaseModel):
    prompt: str = Field(..., max_length=1000)
    negative_prompt: Optional[str] = Field(default="", max_length=1000)
    cfg: float = Field(default=7.0, ge=0.0, le=20.0)
    steps: int = Field(default=50, ge=1, le=250)
    batch_size: int = Field(default=1, ge=1, le=4)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    model: str = Field(default="PrometheusV2-tagFinV0.1")

async def verify_credits(user_id: str, credits_required: int = 1):
    balance = await discord_balance_prompt(user_id, "")
    if balance is None or balance < credits_required:
        raise HTTPException(status_code=402, detail="Insufficient credits")

async def generate_image(request: TextToImageRequest, user_id: str, UUID: str):
    with open(text2img_config, "r") as file:
        workflow = json.load(file)

    cfg_node = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
    workflow = edit_given_nodes_properties(workflow, cfg_node, "cfg", request.cfg)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
    )
    latent_image_nodes = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
    ksampler_nodes = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
    seed = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
    widthnode = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
    heightnode = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
    neg_prompt_nodes = search_for_nodes_with_key(
        "Negative Prompt", workflow, "title", whether_to_use_meta=True
    )
    model_node = search_for_nodes_with_key(
        "Model Checkpoint", workflow, "title", whether_to_use_meta=True
    )
    
    if request.model == "":
        workflow = edit_given_nodes_properties(
            workflow, ksampler_nodes, "sampler_name", "euler_ancestral"
        )
    else:
        workflow = edit_given_nodes_properties(
            workflow, ksampler_nodes, "sampler_name", "dpmpp_3m_sde_gpu"
        )

    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", request.prompt)
    if request.negative_prompt is None:
        request.negative_prompt = "deformed, malformed, worst, bad, embedding:ac_neg1, embedding:ac_neg2"
    workflow = edit_given_nodes_properties(
        workflow, neg_prompt_nodes, "text", request.negative_prompt
    )

    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", request.batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", request.steps)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )

    workflow = edit_given_nodes_properties(
        workflow, widthnode, "width", request.width
    )
    workflow = edit_given_nodes_properties(
        workflow, heightnode, "height", request.height
    )

    if model_node:
        model_name_adjusted = str(request.model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
        )

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, request.model, request.prompt)

    return images

@app.post("/generate")
async def generate_image_endpoint(
    request: TextToImageRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Header(..., description="User ID for credit management")
):
    try:
        # Verify and deduct credits
        await verify_credits(user_id)
        deducted, new_balance = await deduct_credits(user_id, 1)
        if not deducted:
            raise HTTPException(status_code=402, detail="Failed to deduct credits")

        UUID = str(uuid.uuid4())
        images = await generate_image(request, user_id, UUID)
        if not images:
            raise HTTPException(status_code=500, detail="Failed to generate image")

        # For simplicity, we're returning the first image. You might want to handle multiple images differently.
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png", headers={"X-Image-UUID": UUID})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/balance")
async def get_balance(user_id: str = Header(..., description="User ID for credit management")):
    balance = await discord_balance_prompt(user_id, "")
    if balance is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"balance": balance}

@app.post("/recharge")
async def recharge_credits(user_id: str = Header(..., description="User ID for credit management")):
    payment_link = await discord_recharge_prompt(user_id, "")
    if payment_link == "failed":
        raise HTTPException(status_code=500, detail="Failed to generate payment link")
    return {"payment_link": payment_link}

@app.get("/images")
async def get_user_image_history(user_id: str = Header(..., description="User ID for image history")):
    images = await get_user_images(user_id)
    return {"images": images}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "backend_mode": config.get("LOCAL", "TYPE", fallback="cluster")}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    uvicorn.run("sdxl-comfyui-fastapi:app", host="0.0.0.0", port=port, workers=workers)
