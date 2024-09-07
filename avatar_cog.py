import discord
from discord import app_commands
import json
import random
import os
from PIL import Image
import io
import logging

from imageGen import ImageGenerator, search_for_nodes_with_key, edit_given_nodes_properties, upload_image, save_images
from utils import ensure_folder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AvatarCog')

# Load the workflow with error handling
def load_workflow(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        logger.error(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
        return None
    except FileNotFoundError:
        logger.error(f"Workflow file not found: {file_path}")
        return None

workflow = load_workflow("workflow_api.json")

if workflow is None:
    logger.error("Failed to load workflow. Please check the JSON file and correct any formatting errors.")

# Define style presets
STYLE_PRESETS = {
    "Anime": "anime_style.jpg",
    "Realistic": "realistic_style.jpg",
    "Cartoon": "cartoon_style.jpg",
    "Oil Painting": "oil_painting_style.jpg",
    "Watercolor": "watercolor_style.jpg"
}

class AvatarCog:
    def __init__(self, client):
        self.client = client
        if workflow is None:
            raise ValueError("Workflow failed to load. Cannot initialize AvatarCog.")

    @app_commands.command(name="avatar", description="Generate a stylized avatar based on your image and prompt")
    @app_commands.choices(style=[
        app_commands.Choice(name=style, value=style)
        for style in STYLE_PRESETS.keys()
    ])
    async def avatar(self, interaction: discord.Interaction, image: discord.Attachment, style: str, prompt: str):
        await interaction.response.defer()

        # Save the user's image
        user_image_path = f"input/user_{interaction.user.id}.png"
        await image.save(user_image_path)

        # Get the style image path
        style_image_path = f"input/{STYLE_PRESETS[style]}"

        # Modify the workflow
        modified_workflow = self.modify_workflow(workflow, user_image_path, style_image_path, prompt)

        # Generate the image
        try:
            generator = ImageGenerator(host=self.client.config["COMFY_UI"]["SERVER_ADDRESS"])
            await generator.connect()

            # Upload the user's image to the ComfyUI server
            await upload_image(user_image_path, host=generator.host)

            images = await generator.get_images(modified_workflow)
            await generator.close()

            # Save the generated image
            if images:
                output_image = images[0]
                output_path = f"output/avatar_{interaction.user.id}.png"
                output_image.save(output_path)

                # Send the result
                await interaction.followup.send(
                    f"Here's your stylized avatar in {style} style, based on the prompt: '{prompt}'",
                    file=discord.File(output_path)
                )
            else:
                await interaction.followup.send("Sorry, there was an error generating your avatar.")
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            await interaction.followup.send("An error occurred while generating your avatar. Please try again later.")

    def modify_workflow(self, workflow, user_image_path, style_image_path, prompt):
        # Modify the workflow to use the user's image for both avatar nodes
        avatar_1_node = search_for_nodes_with_key("avatar_1", workflow, "title", whether_to_use_meta=True)[0]
        avatar_2_node = search_for_nodes_with_key("avatar_2", workflow, "title", whether_to_use_meta=True)[0]
        style_image_node = search_for_nodes_with_key("style", workflow, "title", whether_to_use_meta=True)[0]

        user_image_filename = os.path.basename(user_image_path)
        workflow = edit_given_nodes_properties(workflow, [avatar_1_node], "image", user_image_filename)
        workflow = edit_given_nodes_properties(workflow, [avatar_2_node], "image", user_image_filename)
        workflow = edit_given_nodes_properties(workflow, [style_image_node], "image", os.path.basename(style_image_path))

        # Modify the prompt in the workflow
        prompt_node = search_for_nodes_with_key("Pos Prompt", workflow, "title", whether_to_use_meta=True)[0]
        workflow = edit_given_nodes_properties(workflow, [prompt_node], "text", prompt)

        # Adjust other parameters as needed
        ksampler_node = search_for_nodes_with_key("KSampler", workflow, "title", whether_to_use_meta=True)[0]
        workflow = edit_given_nodes_properties(workflow, [ksampler_node], "seed", random.randint(0, 1000000000))

        return workflow
