import websockets
import uuid
import json
import random
import urllib.request
import time
import urllib.parse
from PIL import Image as PILImage
from db import init_db
from io import BytesIO
from database_query_same_energy import perform_search
import configparser
import os
import tempfile
import requests
import torch
import ImageReward as reward
import io
from math import ceil, sqrt
import aiofiles
import discord
import logging
import sqlite3
from db import DATABASE_URL
import traceback
from itertools import cycle
from collections import defaultdict
import pathlib
from utils import ensure_folder

IMG_LOGGER = logging.getLogger("Datapulse.imageGen")
IMG_LOGGER.info("Importing imageGen")

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    filename="bot.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)


# Read the configuration
config = configparser.ConfigParser()
config.read("config.properties")

backend_mode = config["LOCAL"]["TYPE"] or "cluster"
server_address = config["LOCAL"]["SERVER_ADDRESS"]
cluster_hosts = config["COMFY_CLUSTER"]["SERVER_ADDRESSES"].split(",")

text2img_config = config["LOCAL_TEXT2IMG"]["CONFIG"]
sigmafiedtext2img_config = config["LOCAL_SIGMAFIED_TEXT2IMG"]["CONFIG"]
img2img_config = config["LOCAL_IMG2IMG"]["CONFIG"]
upscale_config = config["LOCAL_UPSCALE"]["CONFIG"]
style_config = config["LOCAL_STYLE2IMG"]["CONFIG"]
text2imgV3_config = config["LOCAL_TEXT2IMGV3"]["CONFIG"]
describe_config = config["DESCRIBE"]["CONFIG"]
M900_TEXT2IMG_CONFIG = config["900M_TEXT2IMG"]["CONFIG"]
AURAFLOW_TEXT2IMG_CONFIG = config["AURAFLOW_TEXT2IMG"]["CONFIG"]
FLUX_TEXT2IMG_CONFIG = config["FLUX_TEXT2IMG"]["CONFIG"]
KOLORS_TEXT2IMG_CONFIG = config["KOLORS_TEXT2IMG"]["CONFIG"]
GEN_AVATAR_CONFIG = config["GEN_AVATAR"]["CONFIG"]

host_iter = cycle(cluster_hosts)


AVATAR_STYLE_PRESETS = {
    "Anime": "anime_style.jpg",
    "Realistic": "realistic_style.jpg",
    "Cartoon": "cartoon_style.jpg",
    "Oil Painting": "oil_painting_style.jpg",
    "Watercolor": "watercolor_style.jpg"
}


def get_host():
    # global host_iter
    if (backend_mode == 'cluster'):
        return next(host_iter)
    else:
        return server_address


async def save_discord_attachment(attachment: discord.Attachment):
    """Save the attachment from Discord to a specific directory with a unique filename."""
    os.makedirs("input", exist_ok=True)  # Ensure the directory exists
    random_int = random.randint(1000, 9999)
    filename = f"inputimage_{random_int}.png"
    filepath = os.path.join("input", filename)

    # Save the attachment using the discord.py library's save method
    await attachment.save(filepath)
    return filepath


def queue_prompt(prompt, client_id, host=server_address):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request("http://{}/prompt".format(host), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type, host=server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "http://{}/view?{}".format(host, url_values)
    ) as response:
        return response.read()


def get_history(prompt_id, host=server_address):
    with urllib.request.urlopen(
        "http://{}/history/{}".format(host, prompt_id)
    ) as response:
        return json.loads(response.read())


def upload_image(filepath, subfolder=None, folder_type=None, overwrite=False, host=server_address):
    url = f"http://{host}/upload/image"
    files = {"image": open(filepath, "rb")}
    data = {"overwrite": str(overwrite).lower()}
    if subfolder:
        data["subfolder"] = subfolder
    if folder_type:
        data["type"] = folder_type
    response = requests.post(url, files=files, data=data)
    return response.json()


async def save_images(images, user_id, UUID, model, prompt):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    # TODO: fix the fucking uuid count passing through every wormhole known to man problem soon.TM
    print(f"{len(images)} {type(images)}")
    try:
        count = 1
        for image in images:
            u_uuid = f"{UUID}_{count}"
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")  # Corrected save method
            blob = img_byte_arr.getvalue()

            pathlib.Path(f'./generated_images/{u_uuid}.png').write_bytes(blob)

            # Insert the image data into the database
            cursor.execute(
                """
                 INSERT INTO images (user_id, UUID, count, model, prompt)
                 VALUES (?, ?, ?, ?, ?)
                """,
                 (user_id, u_uuid, count, model, prompt),
            )

            count += 1

        # Commit the changes to the database
        conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        print(traceback.format_exc())
        conn.rollback()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        conn.rollback()

    finally:
        cursor.close()
        conn.close()

async def get_prompt_from_database(image_id):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT prompt FROM images WHERE UUID = ?
            """,
            (image_id,),
        )
        return cursor.fetchone()[0]

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        print(traceback.format_exc())
        return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        return None

    finally:
        cursor.close()
        conn.close()

async def get_image_from_database(image_id):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        # Retrieve the image data from the database based on the image ID
        # cursor.execute(
        #     """
        #     SELECT data FROM images WHERE UUID = ?
        # """,
        #     (image_id,),
        # )
        result = pathlib.Path(f'./generated_images/{image_id}.png').read_bytes()
        #result = cursor.fetchone()

        if result:
            image_data = result
            return image_data
        else:
            print(f"Image with UUID {image_id} not found in the database.")
            return None

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        print(traceback.format_exc())
        return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        return None

    finally:
        cursor.close()
        conn.close()

async def read_file_with_retry(file_path, max_retries=5, delay=0.5):
    for attempt in range(max_retries):
        try:
            return pathlib.Path(file_path).read_bytes()
        except FileNotFoundError:
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                raise

async def create_collage(UUID: str, batch_size: int):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    images = []
    result = None

    try:
        for i in range(1, batch_size + 1):
            UUID_count = f"{UUID}_{i}"
            file_path = f'./generated_images/{UUID_count}.png'
            try:
                single_result = await read_file_with_retry(file_path)
                images.append(single_result)
            except FileNotFoundError:
                print(f"No image data found for UUID: {UUID_count} after multiple attempts")
                continue

        if not images:
            print("No images found to create collage")
            return None

        # Handle single image case
        if len(images) == 1:
            result = f'./generated_images/{UUID}.png'
            pathlib.Path(result).write_bytes(images[0])
            return result

        # Open the first image to get its dimensions
        first_image = PILImage.open(BytesIO(images[0]))
        image_width, image_height = first_image.size

        # Calculate optimal grid size
        num_images = len(images)
        cols = ceil(sqrt(num_images))
        rows = ceil(num_images / cols)

        # Create a new blank collage image
        collage_width = cols * image_width
        collage_height = rows * image_height
        collage = PILImage.new("RGB", (collage_width, collage_height))

        # Paste each image onto the collage
        for i, image_data in enumerate(images):
            img = PILImage.open(BytesIO(image_data))
            row = i // cols
            col = i % cols
            collage.paste(img, (col * image_width, row * image_height))
        
        collage_bytes = io.BytesIO()
        collage.save(collage_bytes, format="PNG")
        blob = collage_bytes.getvalue()

        result = f'./generated_images/{UUID}.png'
        pathlib.Path(result).write_bytes(blob)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        return None

    finally:
        cursor.close()
        conn.close()
        return result


class ImageGenerator:
    def __init__(self, host=server_address):
        IMG_LOGGER.info(f"Starting a Generator instance for Host {host}")

        self.client_id = str(uuid.uuid4())
        self.host = host
        self.uri = f"ws://{host}/ws?clientId={self.client_id}"
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.uri)
    
    async def get_output(self, prompt):
        output = None
        if not self.ws:
            await self.connect()
        prompt_id = queue_prompt(prompt, self.client_id, host=self.host)["prompt_id"]
        currently_Executing_Prompt = None
        async for out in self.ws:
            try:
                message = json.loads(out)
                if message["type"] == "execution_start":
                    currently_Executing_Prompt = message["data"]["prompt_id"]
                if (
                    message["type"] == "executing"
                    and prompt_id == currently_Executing_Prompt
                ):
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break
            except ValueError as e:
                IMG_LOGGER.warn(f"Incompatible response from ComfyUI {e}")

        history = get_history(prompt_id, self.host)[prompt_id]

        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            node_string = node_output.get("string")
            if node_string:
                output = node_string[0]

        return output

    async def get_images(self, prompt):
        if not self.ws:
            await self.connect()

        prompt_id = queue_prompt(prompt, self.client_id, host=self.host)["prompt_id"]
        currently_Executing_Prompt = None
        output_images = []
        async for out in self.ws:
            try:
                message = json.loads(out)
                if message["type"] == "execution_start":
                    currently_Executing_Prompt = message["data"]["prompt_id"]
                if (
                    message["type"] == "executing"
                    and prompt_id == currently_Executing_Prompt
                ):
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break
            except ValueError as e:
                print(f"Incompatible response from ComfyUI {e}")

        history = get_history(prompt_id, self.host)[prompt_id]

        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                for image in node_output["images"]:
                    image_data = get_image(
                        image["filename"], image["subfolder"], image["type"], host=self.host
                    )
                    if "final_output" in image["filename"]:
                        pil_image = PILImage.open(BytesIO(image_data))
                        output_images.append(pil_image)

        return output_images

    async def close(self):
        if self.ws:
            await self.ws.close()


def search_for_nodes_with_key(value, workflow, key, whether_to_use_meta=False):
    results = []
    for node_key, node in workflow.items():
        if whether_to_use_meta:
            if "_meta" in node and key in node["_meta"] and node["_meta"][key] == value:
                results.append(node_key)
        else:
            if key in node and node[key] == value:
                results.append(node_key)
            elif key == "class_type" and "class_type" in node and node["class_type"] == value:
                results.append(node_key)
    print(f"Found nodes for {key}={value}: {results}")  # Debug print
    return results

def edit_given_nodes_properties(workflow, chosen_nodes, key, value):
    changes_made = False
    print(f"Editing nodes: {chosen_nodes}, key: {key}, value: {value}")  # Debug print
    for node_key in chosen_nodes:
        if node_key in workflow:
            print(f"Node {node_key} structure: {workflow[node_key]}")  # Debug print
            if "inputs" in workflow[node_key] and key in workflow[node_key]["inputs"]:
                workflow[node_key]["inputs"][key] = value
                changes_made = True
                print(f"Changed node {node_key}")  # Debug print
            else:
                print(f"Warning: Key '{key}' not found in inputs of node {node_key}.")
        else:
            print(f"Warning: Node {node_key} not found in workflow.")

    if not changes_made:
        raise ValueError("Cannot find the chosen nodes or the specified key in their inputs")

    return workflow

async def generate_pixart_900m(
    UUID: str,
    user_id: int,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int
):
    with open(M900_TEXT2IMG_CONFIG, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

    # Update workflow nodes
    positive_prompt_node = search_for_nodes_with_key("Positive Prompt", workflow, "title", whether_to_use_meta=True)
    negative_prompt_node = search_for_nodes_with_key("Negative Prompt", workflow, "title", whether_to_use_meta=True)
    ksampler_node = search_for_nodes_with_key("KSampler", workflow, "title", whether_to_use_meta=True)
    latent_image_node = search_for_nodes_with_key("Empty Latent Image", workflow, "title", whether_to_use_meta=True)

    workflow = edit_given_nodes_properties(workflow, positive_prompt_node, "text", prompt)
    workflow = edit_given_nodes_properties(workflow, negative_prompt_node, "text", negative_prompt)
    workflow = edit_given_nodes_properties(workflow, ksampler_node, "seed", seed)
    workflow = edit_given_nodes_properties(workflow, ksampler_node, "steps", steps)
    workflow = edit_given_nodes_properties(workflow, ksampler_node, "cfg", cfg)
    workflow = edit_given_nodes_properties(workflow, latent_image_node, "width", width)
    workflow = edit_given_nodes_properties(workflow, latent_image_node, "height", height)

    images = await generator.get_images(workflow)
    await generator.close()

    await save_images(images, user_id, UUID, "PixArt900M", prompt)
    return images

async def generate_avatar(
    UUID: str,
    user_id: int,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
    style: str,
    user_image: discord.Attachment
):
    with open(GEN_AVATAR_CONFIG, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

    # Save the user's image
    user_image_path = f"input/user_{UUID}.png"
    await user_image.save(user_image_path)

    # Get the style image path
    style_image_path = f"{style}.jpg"
    
    posprompt = search_for_nodes_with_key(
        "Pos Prompt", workflow, "title", whether_to_use_meta=True
    )
    negprompt = search_for_nodes_with_key(
        "Neg Prompt", workflow, "title", whether_to_use_meta=True
    )
    avatar_1 = search_for_nodes_with_key(
        "avatar_1", workflow, "title", whether_to_use_meta=True
    )
    avatar_2 = search_for_nodes_with_key(
        "avatar_2", workflow, "title", whether_to_use_meta=True
    )
    style = search_for_nodes_with_key(
        "style", workflow, "title", whether_to_use_meta=True
    )
    KSampler= search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
        )
    user_image_filename = os.path.basename(user_image_path)
    
    workflow = edit_given_nodes_properties(
        workflow, negprompt, "text", negative_prompt
    )
    workflow = edit_given_nodes_properties(
        workflow, posprompt, "text", prompt
    )
    workflow = edit_given_nodes_properties(
        workflow, style, "image", style_image_path
    )
    workflow = edit_given_nodes_properties(
        workflow, avatar_1, "image", user_image_filename
    )
    workflow = edit_given_nodes_properties(
        workflow, avatar_2, "image", user_image_filename
    )
    # Upload images
    upload_image(filepath=user_image_path, host=generator.host)
    

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, model, prompt)

    return images


async def sigmafied_image_generation(
    UUID: str,
    user_id: int,
    cfg: float,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
):
   
    with open(sigmafiedtext2img_config, "r") as file:
            workflow = json.load(file)
    cfg_node = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
        )
    workflow = edit_given_nodes_properties(workflow, cfg_node, "cfg", cfg)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

   
    prompt_nodes = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
    )
    ksampler_nodes = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
    )
    seed = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
    )
    widthnode = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
    )
    heightnode = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
    )
    neg_prompt_nodes = search_for_nodes_with_key(
        "Core", workflow, "title", whether_to_use_meta=True
    )


    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "prompt", prompt)
    if negative_prompt is None:
        negative_prompt = "watermark"
    workflow = edit_given_nodes_properties(
        workflow, neg_prompt_nodes, "negative_prompt", negative_prompt
    )

    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "num_inference_steps", steps)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(
        workflow, widthnode, "width", width if width is not None else default_width
    )
    workflow = edit_given_nodes_properties(
        workflow, heightnode, "height", height if height is not None else default_height
    )

    with open("workflow_sigmafied.json", "w") as f:
        json.dump(workflow, f)

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, "sigma", prompt)


async def generate_kolors(
    UUID: str,
    user_id: int,
    cfg: float,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    steps: int,
):
   
    with open(KOLORS_TEXT2IMG_CONFIG, "r") as file:
            workflow = json.load(file)
    cfg_node = search_for_nodes_with_key(
        "KolorsSampler", workflow, "class_type", whether_to_use_meta=False
        )
    workflow = edit_given_nodes_properties(workflow, cfg_node, "cfg", cfg)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

   
    prompt_nodes = search_for_nodes_with_key(
        "KolorsTextEncode", workflow, "class_type", whether_to_use_meta=False
    )
    ksampler_nodes = search_for_nodes_with_key(
        "KolorsSampler", workflow, "class_type", whether_to_use_meta=False
    )
    seed = search_for_nodes_with_key(
        "KolorsSampler", workflow, "class_type", whether_to_use_meta=False
    )
    neg_prompt_nodes = search_for_nodes_with_key(
        "KolorsTextEncode", workflow, "class_type", whether_to_use_meta=False
    )
 
    

    # Modify the prompt dictionary


    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "prompt", prompt)
    workflow = edit_given_nodes_properties(
        workflow, neg_prompt_nodes, "negative_prompt", negative_prompt
    )

    workflow = edit_given_nodes_properties(
        workflow, prompt_nodes, "num_images_per_prompt", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", steps)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(
        workflow, ksampler_nodes, "width", width if width is not None else default_width
    )
    workflow = edit_given_nodes_properties(
        workflow, ksampler_nodes, "height", height if height is not None else default_height
    )

    with open("workflow.json", "w") as f:
        json.dump(workflow, f)

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, "Kolors", prompt)

async def flux1dev(
    UUID: str,
    user_id: int,
    prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
    steps: int,
):

    with open(FLUX_TEXT2IMG_CONFIG, "r") as file:
            workflow = json.load(file)
    generator = ImageGenerator(host=get_host())
    await generator.connect()


    prompt_nodes = search_for_nodes_with_key(
        "CLIP Text Encode (Positive Prompt)", workflow, "title", whether_to_use_meta=True
    )
    latent_image_nodes = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
    ksampler_nodes = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
   
    widthnode = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
    heightnode = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
  


    # Modify the prompt dictionary

    workflow = edit_given_nodes_properties(
            workflow, ksampler_nodes, "scheduler", "simple"
        )


    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)

    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", steps)
    workflow = edit_given_nodes_properties(
        workflow, ksampler_nodes, "seed", random.randint(0, 10000000)
    )
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(
        workflow, widthnode, "width", width if width is not None else default_width
    )
    workflow = edit_given_nodes_properties(
        workflow, heightnode, "height", height if height is not None else default_height
    )
    with open("workflow.json", "w") as f:
        json.dump(workflow, f)

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, model, prompt)

async def generate_images(
    UUID: str,
    user_id: int,
    cfg: float,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
    steps: int,
):
   
    with open(text2img_config, "r") as file:
            workflow = json.load(file)
    cfg_node = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
        )
    workflow = edit_given_nodes_properties(workflow, cfg_node, "cfg", cfg)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

   
    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
    )
    latent_image_nodes = search_for_nodes_with_key(
        "EmptyLatentImage", workflow, "class_type", whether_to_use_meta=False
    )
    ksampler_nodes = search_for_nodes_with_key(
        "KSampler-First", workflow, "title", whether_to_use_meta=True
    )
    seed = search_for_nodes_with_key(
        "KSampler-First", workflow, "title", whether_to_use_meta=True
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

    workflow = edit_given_nodes_properties(
            workflow, ksampler_nodes, "sampler_name", "dpmpp_3m_sde_gpu"
        )

    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    if negative_prompt is None:
        negative_prompt = "deformed, malformed, worst, bad"
    #, embedding:ac_neg1 , embedding:ac_neg2
    workflow = edit_given_nodes_properties(
        workflow, neg_prompt_nodes, "text", negative_prompt
    )

    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", steps)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(
        workflow, widthnode, "width", width if width is not None else default_width
    )
    workflow = edit_given_nodes_properties(
        workflow, heightnode, "height", height if height is not None else default_height
    )
    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = str(model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
        )
    with open("workflow.json", "w") as f:
        json.dump(workflow, f)

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, model, prompt)

async def auraflow(
    UUID: str,
    model: str,
    user_id: int,
    cfg: float,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    steps: int,
):
   
    with open(AURAFLOW_TEXT2IMG_CONFIG, "r") as file:
    	workflow = json.load(file)
    
    cfg_node = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
        )
    workflow = edit_given_nodes_properties(workflow, cfg_node, "cfg", cfg)

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
    

    workflow = edit_given_nodes_properties(
            workflow, ksampler_nodes, "sampler_name", "uni_pc"
        )


    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    if negative_prompt is None:
        negative_prompt = "watermark, worst quality, deformed"
    workflow = edit_given_nodes_properties(
        workflow, neg_prompt_nodes, "text", negative_prompt
    )

    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", steps)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(
        workflow, widthnode, "width", width if width is not None else default_width
    )
    workflow = edit_given_nodes_properties(
        workflow, heightnode, "height", height if height is not None else default_height
    )
    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = str(model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
        )
    with open("workflow.json", "w") as f:
        json.dump(workflow, f)

    images = await generator.get_images(workflow)

    await generator.close()
    await save_images(images, user_id, UUID, model, prompt)



async def style_images(
    UUID: str,
    user_id: int,
    cfg: int,
    prompt: str,
    attachment: discord.Attachment,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    steps: int,
    model: str,
):
    # Save the attachment to a file
    inputname = await save_discord_attachment(attachment)
    filename_without_directory = os.path.basename(inputname)

    # Load the workflow configuration
    with open(style_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator(host=get_host())
    await generator.connect()


    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
    )
    cfg_node = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
    model_node = search_for_nodes_with_key(
        "Model Checkpoint", workflow, "title", whether_to_use_meta=True
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
    neg_prompt_nodes = search_for_nodes_with_key(
        "Negative Prompt", workflow, "title", whether_to_use_meta=True
    )

    if prompt_nodes:
        workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    if neg_prompt_nodes:
        workflow = edit_given_nodes_properties(
            workflow, neg_prompt_nodes, "text", negative_prompt
        )


    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", steps)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    workflow = edit_given_nodes_properties(workflow, cfg_node, "cfg", cfg)
    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = str(model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
        )

    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(
        workflow,
        latent_image_nodes,
        "width",
        width if width is not None else default_width,
    )
    workflow = edit_given_nodes_properties(
        workflow,
        latent_image_nodes,
        "height",
        height if height is not None else default_height,
    )

    upload_image(filepath=inputname, host=generator.host)
    styleimage_nodes = search_for_nodes_with_key(
        "Load Image", workflow, "title", whether_to_use_meta=True
    )
    workflow = edit_given_nodes_properties(
        workflow, styleimage_nodes, "image", filename_without_directory
    )  # Use file path directly
    # we must save the workflow to see what it looks like

    time.sleep(1)
    """
    results = perform_search(prompt)
    
    saved_image_paths = []
    os.makedirs("input_images", exist_ok=True)

    if isinstance(results, list) and all(isinstance(item, bytes) for item in results):
        for i, image_data in enumerate(results, start=1):
            image_name = f"input_images/image_{i}_{random.randint(0, 10000000)}.png"
            with open(image_name, 'wb') as file:
                file.write(image_data)
            saved_image_paths.append(image_name)

    # Load the ImageReward model
    model = reward.load("ImageReward-v1.0")

    # Evaluate the saved images
    with torch.no_grad():
        ranking, rewards = model.inference_rank(prompt, saved_image_paths)

    # Get the best image's path
    best_image_path = saved_image_paths[ranking[0]]
    
    print(f"The best-rated image is at: {best_image_path}")

    
    # Upload the best image
    upload_image(filepath=best_image_path)
    time.sleep(1)
    best_image_filename = os.path.basename(best_image_path)
    workflow = edit_given_nodes_properties(workflow, image_load_nodes, 'image', best_image_filename)
    """
    # Dump workflow into json
    with open("workflow.json", "w") as f:
        json.dump(workflow, f)
        print("Workflow successfully dumped to JSON. to {workflow.json}")

    # Modify batch size

    # if(negative_prompt != None and neg_prompt_nodes[0] != ''): TODO Implement negative prompt
    #   for node in neg_prompt_nodes:
    #       DO STUFF
    # if(rand_seed_nodes[0] != ''):
    #   for node in rand_seed_nodes:
    #       index_of_property = search_for_input_field(workflow["nodes"][node], "seed")
    #       workflow["nodes"][node]["inputs"][index_of_property] = random.randint(0, 100000000)

    images = await generator.get_images(workflow)
    print(f"Images generated: {images}")
    await generator.close()
    # Save the images to the database
    await save_images(images, user_id, UUID, model, prompt)

async def describe_image(uuid, attachment: discord.Attachment, user_id: int):
    workflow = None
    with open(describe_config, "r") as file:
        workflow = json.load(file)

    ensure_folder('./captioned_images')
    stored_image_path = f'./captioned_images/{uuid}.png'
    attachment.save(stored_image_path)

    generator = ImageGenerator(host=get_host())
    await generator.connect()
    upload_image(filepath=stored_image_path)
    styleimage_nodes = search_for_nodes_with_key(
        "Load Image", workflow, "title", whether_to_use_meta=True
    )
    workflow = edit_given_nodes_properties(
        workflow, styleimage_nodes, "image", stored_image_path.split('/')[-1]
    )  # Use file path directly

    with open("workflow.json", 'w') as fp:
        json.dump(workflow, fp)

    caption_output = await generator.get_output(workflow)
    pathlib.Path(f'./captioned_images/{uuid}.txt').write_text(caption_output, 'utf-8')
    IMG_LOGGER.info(f"image described: {caption_output}")
    await generator.close()
    return caption_output
    

async def generate_alternatives(
    UUID: str,
    user_id: int,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
):
    # New UUID for the alternative image batch
    new_UUID = str(uuid.uuid4())

    # get image from db with input passed UUID
    image_data = await get_image_from_database(UUID)
    if image_data is None:
        raise ValueError("Image data not found in the database.")

     ##TODO: Save the image data to a file
    os.makedirs("input", exist_ok=True)
    inputname = f"input/{UUID}.png"
    with open(inputname, "wb") as file:
        file.write(image_data)

    # Load the workflow configuration
    with open(img2img_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
    )
    model_node = search_for_nodes_with_key(
        "Model Checkpoint", workflow, "title", whether_to_use_meta=True
    )
    ksampler_nodes = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
    seed = search_for_nodes_with_key(
        "KSampler", workflow, "class_type", whether_to_use_meta=False
    )
    neg_prompt_nodes = search_for_nodes_with_key(
        "Negative Prompt", workflow, "title", whether_to_use_meta=True
    )

    if prompt_nodes:
        workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    if neg_prompt_nodes:
        workflow = edit_given_nodes_properties(
            workflow, neg_prompt_nodes, "text", negative_prompt
        )

    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", 30)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = str(model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
        )

    print(f"Processing image: {inputname}")
    filename_without_directory = os.path.basename(inputname)
    print(f"Filename without directory: {filename_without_directory}")

    upload_image(filepath=inputname)
    styleimage_nodes = search_for_nodes_with_key(
        "Load Image", workflow, "title", whether_to_use_meta=True
    )
    workflow = edit_given_nodes_properties(
        workflow, styleimage_nodes, "image", filename_without_directory
    )  # Use file path directly

    images = await generator.get_images(workflow)
    print(f"Images generated: {images}")
    await generator.close()
    await save_images(images, user_id, new_UUID, model, prompt)

    return images

import asyncio

async def upscale_image(
    UUID: str,
    user_id: int,
    image: object,
    prompt: str,
    negative_prompt: str,
):
    image_blob = await get_image_from_database(UUID)

    os.makedirs("input", exist_ok=True)

    # Save the image blob to a file
    inputname = f"input/{UUID}.png"
    with open(inputname, "wb") as file:
        file.write(image_blob)

    filename_without_directory = os.path.basename(inputname)
    
   
    # Load the workflow configuration
    with open(upscale_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator(host=get_host())
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
    )
    neg_prompt_nodes = search_for_nodes_with_key(
        "Negative Prompt", workflow, "title", whether_to_use_meta=True
    )

    if prompt_nodes:
        workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    if neg_prompt_nodes:
        workflow = edit_given_nodes_properties(
            workflow, neg_prompt_nodes, "text", negative_prompt
        )


    upload_image(filepath=inputname)
    file_input_nodes = search_for_nodes_with_key(
        "LoadImage", workflow, "class_type", whether_to_use_meta=False
    )
    workflow = edit_given_nodes_properties(
        workflow, file_input_nodes, "image", filename_without_directory
    )  # Use file path directly

    images = await generator.get_images(workflow)
    print(f"Images generated: {images}")
    await generator.close()
    model="Upscale"
    new_uuid=UUID + "_upscaled"
    await save_images(images, user_id, new_uuid, model, prompt)


    image_blob = await get_image_from_database(new_uuid+"_1")


    return image_blob
