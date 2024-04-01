import websockets
import UUID_list
import json
import random
import urllib.request
import time
import urllib.parse
from PIL import Image
from io import BytesIO
from database_query_same_energy import perform_search
import configparser
import os
import tempfile
import requests
import torch
import ImageReward as reward

# Read the configuration
config = configparser.ConfigParser()
config.read('config.properties')
server_address = config['LOCAL']['SERVER_ADDRESS']
text2img_config = config['LOCAL_TEXT2IMG']['CONFIG']
# img2img_config = config['LOCAL_IMG2IMG']['CONFIG']
# upscale_config = config['LOCAL_UPSCALE']['CONFIG']

def queue_prompt(prompt, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())
    
def upload_image(filepath, subfolder=None, folder_type=None, overwrite=False):
    url = f"http://{server_address}/upload/image"
    files = {'image': open(filepath, 'rb')}
    data = {
        'overwrite': str(overwrite).lower()
    }
    if subfolder:
        data['subfolder'] = subfolder
    if folder_type:
        data['type'] = folder_type
    response = requests.post(url, files=files, data=data)
    return response.json()

class ImageGenerator:
    def __init__(self):
        self.client_id = str(UUID_list.UUID_list4())
        self.uri = f"ws://{server_address}/ws?clientId={self.client_id}"
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.uri)

    async def get_images(self, prompt):
        if not self.ws:
            await self.connect()
    
        prompt_id = queue_prompt(prompt, self.client_id)['prompt_id']
        currently_Executing_Prompt = None
        output_images = []
        async for out in self.ws:
            try:
                message = json.loads(out)
                if message['type'] == 'execution_start':
                    currently_Executing_Prompt = message['data']['prompt_id']
                if message['type'] == 'executing' and prompt_id == currently_Executing_Prompt:
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break
            except ValueError as e:
                print("Incompatible response from ComfyUI");
                
        history = get_history(prompt_id)[prompt_id]

        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    if 'final_output' in image['filename']:
                        pil_image = Image.open(BytesIO(image_data))
                        output_images.append(pil_image)

        return output_images

    async def close(self):
        if self.ws:
            await self.ws.close()
            
def search_for_nodes_with_key(value, workflow, key, whether_to_use_meta = False) -> list:
    results = []
    for i, node in enumerate(workflow):
        node = workflow[node]
        if whether_to_use_meta:
            if node["_meta"].get(key) == value:
                results.append(i)
        else:
            if node.get(key) == value:
                results.append(i)

    return results if len(results) > 0 else None

def edit_given_nodes_properties(workflow, chosen_nodes, key, value):
    counter = 0
    changes_made = False
    for workflow_key in workflow.keys():
        if counter in chosen_nodes:
            workflow[workflow_key]['inputs'][key] = value
            changes_made = True
        counter += 1
            
    if not changes_made:
        raise ValueError("Cannot find the chosen nodes")
    
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

async def generate_images(prompt: str,negative_prompt: str,batch_size:int, width:int, height:int):
    with open(text2img_config, 'r') as file:
      workflow = json.load(file)

    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = search_for_nodes_with_key('Positive Prompt', workflow, 'title', whether_to_use_meta=True)
    latent_image_nodes = search_for_nodes_with_key('EmptyLatentImage', workflow, 'class_type', whether_to_use_meta=False)
    ksampler_nodes = search_for_nodes_with_key('KSampler', workflow, 'class_type', whether_to_use_meta=False)
    seed = search_for_nodes_with_key('RemoteChainStart', workflow, 'class_type', whether_to_use_meta=False)
    widthnode=search_for_nodes_with_key('EmptyLatentImage', workflow, 'class_type', whether_to_use_meta=False)
    heightnode=search_for_nodes_with_key('EmptyLatentImage', workflow, 'class_type', whether_to_use_meta=False)
    image_load_nodes = search_for_nodes_with_key('LoadImage', workflow, 'class_type', whether_to_use_meta=False)
    neg_prompt_nodes = search_for_nodes_with_key('Negative Prompt', workflow, 'title', whether_to_use_meta=True)

    # Modify the prompt dictionary
    if(prompt != None and prompt_nodes[0] != ''):
        workflow = edit_given_nodes_properties(workflow, prompt_nodes, 'text', prompt)
    if(negative_prompt != None and neg_prompt_nodes[0] != ''): # TODO Implement negative prompt
        for node in neg_prompt_nodes:
            workflow[node]["inputs"]["text"] = negative_prompt

    workflow = edit_given_nodes_properties(workflow, latent_image_nodes, 'batch_size', batch_size)
    workflow = edit_given_nodes_properties(workflow, ksampler_nodes, 'steps', 50)
    workflow = edit_given_nodes_properties(workflow, seed, 'seed', random.randint(0, 10000000))
    default_width = 1024
    default_height = 1024

    # Modify the workflow nodes for width and height with provided values or defaults
    workflow = edit_given_nodes_properties(workflow, widthnode, 'width', width if width is not None else default_width)
    workflow = edit_given_nodes_properties(workflow, heightnode, 'height', height if height is not None else default_height)

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

async def generate_alternatives(image: Image.Image, prompt: str, negative_prompt: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
      image.save(temp_file, format="PNG")
      temp_filepath = temp_file.name

    # Upload the temporary file using the upload_image method
    response_data = upload_image(temp_filepath)
    filename = response_data['name']
    with open(img2img_config, 'r') as file:
      workflow = json.load(file)
      
    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = config.get('LOCAL_IMG2IMG', 'PROMPT_NODES').split(',')
    neg_prompt_nodes = config.get('LOCAL_IMG2IMG', 'NEG_PROMPT_NODES').split(',')
    rand_seed_nodes = config.get('LOCAL_IMG2IMG', 'RAND_SEED_NODES').split(',') 
    file_input_nodes = config.get('LOCAL_IMG2IMG', 'FILE_INPUT_NODES').split(',') 

    if(prompt != None and prompt_nodes[0] != ''):
      for node in prompt_nodes:
          workflow[node]["inputs"]["text"] = prompt
    if(negative_prompt != None and neg_prompt_nodes[0] != ''):
      for node in neg_prompt_nodes:
          workflow[node]["inputs"]["text"] = negative_prompt
    if(rand_seed_nodes[0] != ''):
      for node in rand_seed_nodes:
          workflow[node]["inputs"]["seed"] = random.randint(0,999999999999999)
    if(file_input_nodes[0] != ''):
      for node in file_input_nodes:
          workflow[node]["inputs"]["image"] = filename

    images = await generator.get_images(workflow)
    await generator.close()

    return images

async def upscale_image(image: Image.Image, prompt: str,negative_prompt: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
      image.save(temp_file, format="PNG")
      temp_filepath = temp_file.name

    # Upload the temporary file using the upload_image method
    response_data = upload_image(temp_filepath)
    filename = response_data['name']
    with open(upscale_config, 'r') as file:
      workflow = json.load(file)

    generator = ImageGenerator()
    await generator.connect()

    prompt_nodes = config.get('LOCAL_UPSCALE', 'PROMPT_NODES').split(',')
    neg_prompt_nodes = config.get('LOCAL_UPSCALE', 'NEG_PROMPT_NODES').split(',')
    rand_seed_nodes = config.get('LOCAL_UPSCALE', 'RAND_SEED_NODES').split(',') 
    file_input_nodes = config.get('LOCAL_UPSCALE', 'FILE_INPUT_NODES').split(',') 

    # Modify the prompt dictionary
    if(prompt != None and prompt_nodes[0] != ''):
      for node in prompt_nodes:
          workflow[node]["inputs"]["text"] = prompt
    if(negative_prompt != None and neg_prompt_nodes[0] != ''):
      for node in neg_prompt_nodes:
          workflow[node]["inputs"]["text"] = negative_prompt
    if(rand_seed_nodes[0] != ''):
      for node in rand_seed_nodes:
          workflow[node]["inputs"]["seed"] = random.randint(0,999999999999999)
    if(file_input_nodes[0] != ''):
      for node in file_input_nodes:
          workflow[node]["inputs"]["image"] = filename

    images = await generator.get_images(workflow)
    await generator.close()

    return images[0]