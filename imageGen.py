import websockets
import uuid
import json
import random
import urllib.request
import time
import urllib.parse
from PIL import Image as PILImage
from db import  init_db
from io import BytesIO
from database_query_same_energy import perform_search
import configparser
import os
import tempfile
import requests
import torch
import ImageReward as reward
import io
from sqlalchemy.future import select
from math import ceil, sqrt
import aiofiles
import discord
import logging
import sqlite3
from db import DATABASE_URL

# Read the configuration
config = configparser.ConfigParser()
config.read("config.properties")
server_address = config["LOCAL"]["SERVER_ADDRESS"]
text2img_config = config["LOCAL_TEXT2IMG"]["CONFIG"]
img2img_config = config["LOCAL_IMG2IMG"]["CONFIG"]
upscale_config = config["LOCAL_UPSCALE"]["CONFIG"]
style_config = config["LOCAL_STYLE2IMG"]["CONFIG"]
lora_config = config["LOCAL_LORA"]["CONFIG"]

async def save_discord_attachment(attachment: discord.Attachment):
    """Save the attachment from Discord to a specific directory with a unique filename."""
    os.makedirs('input', exist_ok=True)  # Ensure the directory exists
    random_int = random.randint(1000, 9999)
    filename = f"inputimage_{random_int}.png"
    filepath = os.path.join('input', filename)

    # Save the attachment using the discord.py library's save method
    await attachment.save(filepath)
    return filepath

def queue_prompt(prompt, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(
        "http://{}/view?{}".format(server_address, url_values)
    ) as response:
        return response.read()


def get_history(prompt_id):
    with urllib.request.urlopen(
        "http://{}/history/{}".format(server_address, prompt_id)
    ) as response:
        return json.loads(response.read())


def upload_image(filepath, subfolder=None, folder_type=None, overwrite=False):
    url = f"http://{server_address}/upload/image"
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

    try:
        count = 1
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")  # Corrected save method
            blob = img_byte_arr.getvalue()

            # Save the image to a file and use the file path as the URL
            image_path = f'output/image_{count}.png'
            image.save(image_path)

            # Insert the image data into the database
            cursor.execute("""
                INSERT INTO images (url, data, user_id, UUID, count, model, prompt)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_path, blob, user_id, UUID, count, model, prompt))

            count += 1

        # Commit the changes to the database
        conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        conn.rollback()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        conn.rollback()

    finally:
        cursor.close()
        conn.close()


async def get_image_from_database(image_id):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        # Retrieve the image data from the database based on the image ID
        cursor.execute("""
            SELECT data FROM images WHERE id = ?
        """, (image_id,))
        result = cursor.fetchone()

        if result:
            image_data = result[0]
            return image_data
        else:
            print(f"Image with ID {image_id} not found in the database.")
            return None

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        cursor.close()
        conn.close()


async def create_collage(UUID: str):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM images
            WHERE UUID = ?
            ORDER BY count
        """, (UUID,))
        images = cursor.fetchall()

        # Open the first image to get its width and height
        first_image = PILImage.open(BytesIO(images[0][2]))  # Assuming image data is stored in the 3rd column
        image_width, image_height = first_image.width, first_image.height

        # Determine the dimensions of the collage
        num_images = len(images)
        rows = cols = ceil(sqrt(num_images))
        collage_width, collage_height = cols * image_width, rows * image_height

        # Create a new blank collage image
        collage = PILImage.new("RGB", (collage_width, collage_height))

        # Paste each image onto the collage
        for i, image in enumerate(images):
            img = PILImage.open(BytesIO(image[2]))  # Assuming image data is stored in the 3rd column
            row, col = i // cols, i % cols
            collage.paste(img, (col * image_width, row * image_height))

        # Save the collage to a file
        collage_path = f"collageimages/collage_{UUID}.png"
        collage.save(collage_path)

        return collage_path

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        cursor.close()
        conn.close()


class ImageGenerator:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.uri = f"ws://{server_address}/ws?clientId={self.client_id}"
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.uri)

    async def get_images(self, prompt):
        if not self.ws:
            await self.connect()

        prompt_id = queue_prompt(prompt, self.client_id)["prompt_id"]
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
                print("Incompatible response from ComfyUI")

        history = get_history(prompt_id)[prompt_id]

        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                for image in node_output["images"]:
                    image_data = get_image(
                        image["filename"], image["subfolder"], image["type"]
                    )
                    if "final_output" in image["filename"]:
                        pil_image = PILImage.open(BytesIO(image_data))
                        output_images.append(pil_image)

        return output_images

    async def close(self):
        if self.ws:
            await self.ws.close()


def search_for_nodes_with_key(value, workflow, key, whether_to_use_meta=False) -> list:
    results = []
    for node_key, node in workflow.items():
        node_content = (
            node["_meta"] if whether_to_use_meta and "_meta" in node else node
        )
        if node_content.get(key) == value:
            results.append(node_key)
    return results


def edit_given_nodes_properties(workflow, chosen_nodes, key, value):
    changes_made = False
    for node_key in chosen_nodes:
        if node_key in workflow:
            # Split the key by '.' to handle nested keys
            keys = key.split('.')
            item = workflow[node_key]["inputs"]
            for k in keys[:-1]:
                item = item.get(k, {})
            if keys[-1] in item:
                item[keys[-1]] = value
                changes_made = True
            else:
                print(f"Warning: Key {keys[-1]} not found in node {node_key}.")
        else:
            print(f"Warning: Node {node_key} not found in workflow.")

    if not changes_made:
        raise ValueError("Cannot find the chosen nodes or keys")

    return workflow


async def evaluate_images_with_image_reward(prompt: str, img_list: list):
    """
    Evaluates a list of image file paths using ImageReward and returns the best image path.
    :param prompt: The prompt used for generating the images.
    :param img_list: A list of paths to the generated images.
    :return: Path to the best-rated image according to ImageReward.
    """
    model = reward.load("ImageReward-v1.0")
    with torch.no_grad():
        ranking, rewards = model.inference_rank(prompt, img_list)
    best_image_idx = ranking[0] - 1  # Adjust if your ranking starts from 0
    best_image_path = img_list[best_image_idx]
    return best_image_path



async def lora_images(
    UUID: str,
    user_id: int,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
    lora: str,
):

    # Your existing logic to prepare for image generation...
    with open(lora_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator()
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
    lora_node = search_for_nodes_with_key(
        "Lora Loader", workflow, "title", whether_to_use_meta=True
    )

    
    lora += ".safetensors"
    workflow = edit_given_nodes_properties(workflow, lora_node, "lora_name.content", lora)
        

    if model == "AnimeP":
        workflow = edit_given_nodes_properties( workflow ,ksampler_nodes, "sampler_name", "euler_ancestral")

    # Modify the prompt dictionary
   
    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    
    workflow = edit_given_nodes_properties(
            workflow, neg_prompt_nodes, "text", negative_prompt
        )

    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", 50)
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



async def generate_images(
    UUID: str,
    user_id: int,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
    lora: str,
):

    # Your existing logic to prepare for image generation...
    with open(text2img_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator()
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
    lora_node = search_for_nodes_with_key(
        "LoRA", workflow, "title", whether_to_use_meta=True
    )

    if lora is not None:
        lora += ".safetensors"
        workflow = edit_given_nodes_properties(workflow, lora_node, "lora_name", lora)
        workflow = edit_given_nodes_properties(workflow, lora_node, "strength_model", "1")
        workflow = edit_given_nodes_properties(workflow, lora_node, "strength_clip", "1")

    if model == "AnimeP":
        workflow = edit_given_nodes_properties( workflow ,ksampler_nodes, "sampler_name", "euler_ancestral")

    # Modify the prompt dictionary
   
    workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    
    workflow = edit_given_nodes_properties(
            workflow, neg_prompt_nodes, "text", negative_prompt
        )

    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", 50)
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


"""
async def generate_images(prompt: str,negative_prompt: str,batch_size:int, width:int, height:int, model:str):
    with open(text2img_config, 'r') as file:
      workflow = json.load(file)

    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key('Positive Prompt', workflow, 'title', whether_to_use_meta=True)
    latent_image_nodes = search_for_nodes_with_key('EmptyLatentImage', workflow, 'class_type', whether_to_use_meta=False)
    ksampler_nodes = search_for_nodes_with_key('KSampler', workflow, 'class_type', whether_to_use_meta=False)
    seed = search_for_nodes_with_key('KSampler', workflow, 'class_type', whether_to_use_meta=False)
    widthnode=search_for_nodes_with_key('EmptyLatentImage', workflow, 'class_type', whether_to_use_meta=False)
    heightnode=search_for_nodes_with_key('EmptyLatentImage', workflow, 'class_type', whether_to_use_meta=False)
    neg_prompt_nodes = search_for_nodes_with_key('Negative Prompt', workflow, 'title', whether_to_use_meta=True)
    model_node = search_for_nodes_with_key('Model Checkpoint', workflow, 'title', whether_to_use_meta=True)

    # Modify the prompt dictionary
    if(prompt != None and prompt_nodes[0] != ''):
        workflow = edit_given_nodes_properties(workflow, prompt_nodes, 'text', prompt)
    if neg_prompt_nodes:
        workflow = edit_given_nodes_properties(workflow, neg_prompt_nodes, 'text', negative_prompt)

    workflow = edit_given_nodes_properties(workflow, latent_image_nodes, 'batch_size', batch_size)
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, 'steps', 50)
    workflow = edit_given_nodes_properties(workflow, seed, 'seed', random.randint(0, 10000000))
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(workflow, widthnode, 'width', width if width is not None else default_width)
    workflow = edit_given_nodes_properties(workflow, heightnode, 'height', height if height is not None else default_height)
    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = model + '.safetensors'
        workflow = edit_given_nodes_properties(workflow, model_node, 'ckpt_name', model_name_adjusted)
    
    
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
    
    # Dump workflow into json
    with open("workflow.json", 'w') as f:
        json.dump(workflow, f)
    
    # Modify batch size

    # if(negative_prompt != None and neg_prompt_nodes[0] != ''): TODO Implement negative prompt
    #   for node in neg_prompt_nodes:
    #       DO STUFF
    # if(rand_seed_nodes[0] != ''):
    #   for node in rand_seed_nodes:
    #       index_of_property = search_for_input_field(workflow["nodes"][node], "seed")
    #       workflow["nodes"][node]["inputs"][index_of_property] = random.randint(0, 100000000)

    images = await generator.get_images(workflow)
    await generator.close()

    return images 
"""


# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    filename="bot.log",
    filemode="a",
    format="%(name)s - %(levelname)s - %(message)s",
)


async def style_images(
    UUID: str,
    user_id: int,
    prompt: str,
    attachment: discord.Attachment,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
):
    
    # Save the attachment to a file
    inputname = await save_discord_attachment(attachment)
    filename_without_directory = os.path.basename(inputname)

    # Load the workflow configuration
    with open(style_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
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
    if model == "AnimeP":
        workflow = edit_given_nodes_properties( workflow ,ksampler_nodes, "sampler_name", "euler_ancestral")
    if prompt_nodes:
        workflow = edit_given_nodes_properties(workflow, prompt_nodes, "text", prompt)
    if neg_prompt_nodes:
        workflow = edit_given_nodes_properties(
            workflow, neg_prompt_nodes, "text", negative_prompt
        )

    """
    if(negative_prompt != None and neg_prompt_nodes[0] != ''): # TODO Implement negative prompt
        for node in neg_prompt_nodes:
            workflow[node]["inputs"]["text"] = negative_prompt
    """
    workflow = edit_given_nodes_properties(
        workflow, latent_image_nodes, "batch_size", batch_size
    )
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", 50)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
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

    upload_image(filepath=inputname)
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


async def generate_alternatives(
    UUID: str,
    user_id: int,
    index: int,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    width: int,
    height: int,
    model: str,
):
    #get image from db
    inputname = await get_image_from_database(UUID)

    # Load the workflow configuration
    with open(img2img_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
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
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, "steps", 50)
    workflow = edit_given_nodes_properties(
        workflow, seed, "seed", random.randint(0, 10000000)
    )
    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = str(model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
        )

    default_width = 1024
    default_height = 1024
    if model == "AnimeP":
        workflow = edit_given_nodes_properties( workflow ,ksampler_nodes, "sampler_name", "euler_ancestral")
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
    await save_images(images, user_id, UUID, model, prompt)
    
    return images


async def upscale_image(
    UUID: str,
    user_id: int,
    image: PILImage.Image,
    prompt: str,
    negative_prompt: str,
    model: str,
):
    
    # Ensure the 'input' directory exists
    inputname = await save_discord_attachment(attachment)

    # Load the workflow configuration
    with open(upscale_config, "r") as file:
        workflow = json.load(file)

    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key(
        "Positive Prompt", workflow, "title", whether_to_use_meta=True
    )
    model_node = search_for_nodes_with_key(
        "Model Checkpoint", workflow, "title", whether_to_use_meta=True
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

    if model_node:
        # Before setting the model, ensure the model name is adjusted to remove ".safetensors" if present
        model_name_adjusted = str(model) + ".safetensors"
        workflow = edit_given_nodes_properties(
            workflow, model_node, "ckpt_name", model_name_adjusted
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
    await save_images(images, user_id, UUID, model, prompt)
    
    return images[0]


