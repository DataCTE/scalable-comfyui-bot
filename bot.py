import logging
logging.basicConfig()
from imageGen import generate_avatar, AVATAR_STYLE_PRESETS, generate_pixart_900m
from imageGen import generate_kolors, generate_images
import discord
import discord.ext
from discord import app_commands
import configparser
from PIL import Image
from datetime import datetime
from db import init_db
from imageGen import *
from discord.app_commands import Choice
import uuid
from payment_service import *
from utils import config
from typing import Optional
from stripe_integration import *
import functools
import sqlite3
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import asyncio
import traceback
from discord.ext import tasks
from stripe_integration import verify_payment_links_job
import time
import replicate
import json

# setting up the bot
config = configparser.ConfigParser()
config.read("config.properties")
TOKEN = config["DISCORD"]["TOKEN"]
IMAGE_SOURCE = config["IMAGE"]["SOURCE"]
stripe_api_key = config["STRIPE"]["API_KEY"]
stripe_product_id = config.get("STRIPE", "PRODUCT_ID")
# Dictionary to store conversation history and system prompt for each user
user_contexts = {}

# Default system prompt
default_system_prompt = "You are a helpful AI assistant."

# Replicate API setup
os.environ["REPLICATE_API_TOKEN"] = config["REPLICATE"]["API_TOKEN"]

#stripe.api_key = stripe_api_key
intents = discord.Intents.all()
intents.members = True  # Enable the members intent
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)


if IMAGE_SOURCE == "LOCAL":
    server_address = config.get("LOCAL", "SERVER_ADDRESS")
    from imageGen import generate_images, upscale_image, generate_alternatives, get_image_from_database, get_prompt_from_database, describe_image, sigmafied_image_generation



async def extract_index_from_id(button_id):
    try:
        # Extracting numeric part while assuming it might contain non-numeric characters
        numeric_part = "".join([char for char in button_id if char.isdigit()])
        if not numeric_part:
            return None
        return int(numeric_part) - 1
    except ValueError:
        return None


class ImageButton(discord.ui.Button):
    def __init__(self, custom_id, emoji, row, col, callback):
        super().__init__(
            label=custom_id,
            emoji=emoji,
            row=row,
            style=discord.ButtonStyle.grey,
        )
        self.inner_callback = callback

    async def callback(self, interaction: discord.Interaction):
        await self.inner_callback(interaction)

class Buttons(discord.ui.View):
    def __init__(
        self,
        prompt,
        negative_prompt,
        UUID,
        user_id,
        model,
        images,
        timeout=10000000000000,
        batch_size: int = 4,
    ):
        super().__init__(timeout=timeout)
        self.UUID = UUID  # Store the UUID
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.user_id = user_id
        self.model = model
        self.images = images
        self.batch_size = batch_size

        # Limit the batch size to 8
        batch_size = min(batch_size, 8)

        # Inside the Buttons class __init__ method, adjust the button callbacks setup
        reroll_row = 0  # Reroll buttons will be in the first row
        upscale_row = 1  # Upscale buttons will be in the second row

        for idx in range(batch_size):
            count = idx + 1
            u_uuid = f"{self.UUID}_{count}"
            col = idx  # Each button gets its own column

            # Use functools.partial to correctly prepare the callback with necessary arguments
            reroll_callback = functools.partial(self.reroll_image, u_uuid=u_uuid)
            btn = ImageButton(f"V{count}", "♻️", reroll_row, col, reroll_callback)
            self.add_item(btn)

            if idx < 4:  # Only add upscale button for the first 4 images
                upscale_callback = functools.partial(self.upscale_image, u_uuid=u_uuid)
                btn = ImageButton(f"U{count}", "⬆️", upscale_row, col, upscale_callback)
                self.add_item(btn)
                
    @classmethod
    async def create(cls, prompt, negative_prompt, UUID, user_id, model, batch_size):
        return cls(prompt, negative_prompt, UUID, user_id, model, batch_size)

    async def reroll_image(self, interaction: discord.Interaction, u_uuid):
        try:
            await interaction.response.send_message(f"{interaction.user.mention} asked me to re-imagine the image, this shouldn't take too long...")
            # await interaction.followup.send(f"{interaction.user.mention} asked me to re-imagine the image, this shouldn't take too long...")
            # await interaction.channel.send(
            #     f"{interaction.user.mention} asked me to re-imagine the image, this shouldn't take too long..."
            # )
                # credit check
            user_id = interaction.user.id
            username = interaction.user.name
            user_credits = await discord_balance_prompt(user_id, username)
            if user_credits is None:
                # add them to the db
                username = interaction.user.name
                user_id = interaction.user.id
                create_DB_user(user_id, username)
            elif user_credits < 5:  # Assuming 5 credits are needed
                payment_link = await discord_recharge_prompt(
                    interaction.user.name, self.user_id
                )
                if payment_link == "failed":
                    await interaction.response.send_message(
                        "Failed to create payment link or payment itself failed. Please try again later.",
                        ephemeral=True,
                    )
                    return
                await interaction.response.send_message(
                    f"You don't have enough credits. Please recharge your account: {payment_link}",
                    ephemeral=True,
                )

            # Retrieve the prompt from the database
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()

            try:
                prompt = await get_prompt_from_database(image_id=u_uuid) # yis

                # Generate a new image with the retrieved prompt
                images = await generate_alternatives(
                    UUID=u_uuid,
                    user_id=interaction.user.id,
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    batch_size=self.batch_size,
                    width=1024,
                    height=1024,
                    model=self.model,
                )
                await save_images(UUID=u_uuid, images=images, user_id=interaction.user.id, model=self.model, prompt=prompt)

                # Create a new collage for the re-rolled image
                collage_path = await create_collage(UUID=u_uuid, batch_size=self.batch_size)
                

                # Construct the final message with user mention
                final_message = f'{interaction.user.mention} asked me to re-imagine the image, here is what I imagined for them. "{prompt}", "{self.model}"'
                await interaction.delete_original_response()
                await interaction.channel.send( 
                    content=final_message,
                    file=discord.File(fp=collage_path, filename="collage.png"),
                    view=Buttons(
                        images=images,
                        prompt=self.negative_prompt,
                        UUID=u_uuid,
                        user_id=interaction.user.id,
                        model=self.model,
                        negative_prompt=self.negative_prompt,
                    ),
                )
                # after successful reroll, deduct credits
                # amount = user_credits - 5
                # await deduct_credits(user_id, amount)

            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")
                print(traceback.format_exc())

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print(traceback.format_exc())

            finally:
                cursor.close()
                conn.close()

        except discord.errors.InteractionResponded as e:
            return print(f"{e}: interaction failed")

    async def upscale_image(self, interaction: discord.Interaction, u_uuid):
        await interaction.response.defer()  # Acknowledge the interaction

        try:

            await interaction.channel.send(
                f"{interaction.user.mention} asked me to upscale the image, this shouldn't take too long..."
            )
                # check credits
            user_id = interaction.user.id
            username = interaction.user.name
            user_credits = await discord_balance_prompt(user_id, username)
            if user_credits is None:
                # add them to the db
                username = interaction.user.name
                user_id = interaction.user.id
                create_DB_user(user_id, username)

            elif user_credits < 5:  # Assuming 5 credits are needed
                payment_link = await discord_recharge_prompt(
                    interaction.user.name, self.user_id
                )
                if payment_link == "failed":
                    await interaction.response.send_message(
                        "Failed to create payment link or payment itself failed. Please try again later.",
                        ephemeral=True,
                    )
                    return
                await interaction.response.send_message(
                    f"You don't have enough credits. Please recharge your account: {payment_link}",
                    ephemeral=True,
                )
                return

            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()

            try:
                image = await get_image_from_database(image_id=u_uuid)  # Await the coroutine
                

                # Upscale image logic assumed to be defined elsewhere
                upscaled_image = await upscale_image(
                    image=image, prompt=self.prompt, negative_prompt=self.negative_prompt, user_id=self.user_id, UUID=u_uuid
                )

                # os.makedirs("input", exist_ok=True)
                # inputname = f"input/{u_uuid}.png"
                # with open(inputname, "wb") as file:
                #     file.write(upscaled_image)
                #convert to bytes
              

                final_message = (
                    f"{interaction.user.mention} here is your upscaled image"
                )
                await interaction.channel.send( 
                    content=final_message,
                    file=discord.File(
                        fp=io.BytesIO(image), filename="upscaled_image.png"
                    ),
                )
                # deduct credits
                #amount = user_credits - 5
                #print(amount)
                #await deduct_credits(user_id, amount)

            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")
                print(traceback.format_exc())

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print(traceback.format_exc())

            finally:
                cursor.close()
                conn.close()

        except discord.errors.InteractionResponded:
            print("Interaction already responded")

def is_allowed_channel(interaction: discord.Interaction) -> bool:
    # Disallow DMs
    if isinstance(interaction.channel, discord.DMChannel):
        return False
    
    # Disallow channels named "general"
    if interaction.channel.name.lower() == "general":
        return False
    
    return True

def allowed_channel():
    async def predicate(interaction: discord.Interaction):
        if not is_allowed_channel(interaction):
            await interaction.response.send_message("This command is not allowed in this channel.", ephemeral=True)
            return False
        return True
    return app_commands.check(predicate)

@client.event
async def on_ready():
    init_db()  # Initialize DB
    await tree.sync()

    @tree.error
    async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.errors.CheckFailure):
            # This will catch any check failures, including our custom channel check
            await interaction.response.send_message("This command is not allowed here.", ephemeral=True)
        else:
            # Handle other types of errors
            await interaction.response.send_message("An error occurred while processing the command.", ephemeral=True)

    print(f"Logged in as {client.user.name} ({client.user.id})")

@tree.command(name="chat", description="Chat with the AI assistant")
@app_commands.describe(
    message="Your message to the AI",
    wipe_memory="Set to True to wipe conversation history before this message",
    system_prompt="Set a custom system prompt for this conversation"
)
async def chat(
    interaction: discord.Interaction, 
    message: str, 
    wipe_memory: bool = False,
    system_prompt: Optional[str] = None
):
    global user_contexts
    user_id = str(interaction.user.id)
    
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "messages": [],
            "system_prompt": default_system_prompt
        }
    
    if wipe_memory:
        user_contexts[user_id]["messages"] = []
    
    if system_prompt:
        user_contexts[user_id]["system_prompt"] = system_prompt
    
    user_contexts[user_id]["messages"].append({"role": "user", "content": message})
    
    full_prompt = f"{user_contexts[user_id]['system_prompt']}\n\nConversation history:\n"
    for msg in user_contexts[user_id]["messages"]:
        full_prompt += f"{msg['role']}: {msg['content']}\n"
    full_prompt += "assistant:"
    
    await interaction.response.defer()
    
    response = ""
    for event in replicate.stream(
        "meta/meta-llama-3.1-405b-instruct",
        input={
            "prompt": full_prompt,
            "max_tokens": 2048
        }
    ):
        response += event
    
    user_contexts[user_id]["messages"].append({"role": "assistant", "content": response})
    
    # Format the response
    formatted_response = format_response(response)
    
    # Create an embed for the response
    embed = discord.Embed(
        title="AI Assistant Response",
        description=formatted_response,
        color=discord.Color.blue()
    )
    embed.set_footer(text=f"Requested by {interaction.user.name}")
    
    await interaction.followup.send(embed=embed)

def format_response(response):
    # Split the response into paragraphs
    paragraphs = response.split('\n\n')
    
    formatted = []
    for para in paragraphs:
        # Check if the paragraph is a list
        if para.startswith(('- ', '• ', '* ')):
            formatted.append(para)
        elif para.strip().startswith(('1. ', '2. ', '3. ')):
            formatted.append(para)
        else:
            # Wrap normal paragraphs in quotes
            formatted.append(f"> {para}")
    
    # Join the formatted paragraphs
    formatted_response = '\n\n'.join(formatted)
    
    # Apply additional formatting
    formatted_response = formatted_response.replace('**', '**')  # Bold
    formatted_response = formatted_response.replace('*', '*')    # Italic
    formatted_response = formatted_response.replace('`', '`')    # Inline code
    
    # Check if there are any code blocks and format them
    if '```' in formatted_response:
        lines = formatted_response.split('\n')
        in_code_block = False
        for i, line in enumerate(lines):
            if line.startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    # Start of code block, add language if not specified
                    if len(line) == 3:
                        lines[i] = '```python'
            elif in_code_block:
                # Indent code within code blocks
                lines[i] = '    ' + line
        formatted_response = '\n'.join(lines)
    
    return formatted_response

@tree.command(name="imagine", description="Generate an image based on input text")
@allowed_channel()
@app_commands.describe(prompt="Prompt for the image being generated")
@app_commands.describe(negative_prompt="Prompt for what you want to steer the AI away from")
@app_commands.describe(batch_size="Number of images to generate" )
@app_commands.describe(width="width of the image")
@app_commands.describe(height="height of the image" )
@app_commands.describe(cfg="cfg to use")
@app_commands.describe(attachment="attachment to use")
@app_commands.describe(steps="steps to use")
@app_commands.choices(model=[
    Choice(name="proteusV0.5", value="proteusV0.5"),
    Choice(name="Prometheus", value="Prometheus"),
    Choice(name="PrometheusV2_beta", value="PrometheusV2_beta"),
    Choice(name="Proteus-Prometheus", value="Prometheus"),
])
@app_commands.describe(lora="Choose the lora to use")
@app_commands.choices(lora=[
    Choice(name="Detail", value="tweak-detail-xl"),
    Choice(name="SythAnimeV2", value="AnimeSythenticV0.2"),
    Choice(name="Artistic", value="xl_more_art-full_v1"),
])
async def imagine(
    interaction: discord.Interaction, 
    prompt: str, 
    negative_prompt: str = None,
    batch_size: int = 4,
    width: int = 1024,
    height: int = 1024,
    model: str = "PrometheusV2_beta",
    attachment: discord.Attachment = None, 
    lora: str = None,
    cfg: float = 7.0,
    steps: int = 50
):
## TODO: package parameters into a dataclass and build functions around it.
   
    username = interaction.user.name
    user_id = interaction.user.id

    user_credits = await discord_balance_prompt(user_id, username)

    if batch_size > 4:
        await interaction.response.send_message(
            "The maximum batch size is 4. Please try again with a smaller batch size.",
            ephemeral=True,
        )
        return

    if user_credits is None:
        # Handle case where user credits couldn't be retrieved
        create_DB_user(user_id, username)

    user_credits = await discord_balance_prompt(user_id, username)

    if user_credits < 10:  # Assuming 5 credits are needed
        payment_link = await discord_recharge_prompt(username, user_id)
        if payment_link == "failed":
            await interaction.response.send_message(
                "Failed to create payment link or payment itself failed. Please try again later.",
                ephemeral=True,
            )

        await interaction.response.send_message(
            f"You don't have enough credits. Please recharge your account: {payment_link}",
            ephemeral=True,
        )
        await interaction.response.defer(ephemeral=True)
    else:
        
        await interaction.response.defer(ephemeral=False)
        
        

    UUID = str(uuid.uuid4())  # Generate unique hash for each image

    if attachment:
        # save input image

        await style_images(
            UUID=UUID,
            cfg=cfg,
            steps=steps,
            user_id=interaction.user.id,
            prompt=prompt,
            attachment=attachment,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            model=model,
        )
    else:
        await generate_images(
            UUID=UUID,
            cfg=cfg,
            steps=steps,
            user_id=interaction.user.id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            model=model,
            lora=lora,
        )

    collage_blob = await create_collage(UUID, batch_size)


    buttons_view = await Buttons.create(
        prompt=prompt,
        negative_prompt=negative_prompt,
        UUID=UUID,
        user_id=interaction.user.id,
        model=model,
        batch_size=batch_size,
    )
   
   
    file = discord.File(collage_blob, filename="collage.png")
    final_message = f"{interaction.user.mention}, here is what I imagined for you with ```{prompt}, {model}```"

    await interaction.followup.send(
        content=final_message, file=file, view=buttons_view, ephemeral=False
    )


    #amount = user_credits - 10
    #print(amount)
    #await deduct_credits(user_id, amount)

    return UUID


@tree.command(name="describe", description="Describe an image")
@app_commands.describe(image="Image to describe")
async def describe(interaction: discord.Interaction, image: discord.Attachment):
    if image:
        # Retrieve the image data from the attachment
        image_data = await image.read()

        # Open the image using Pillow
        image = Image.open(BytesIO(image_data))
        imguuid = uuid.uuid4()

        # Generate the caption using the pre-trained model
        await interaction.response.send_message("🧐 Analyzing your image!")
        caption = await describe_image(imguuid, image, interaction.user.id)
        await interaction.followup.send(content=f"<@{interaction.user.id}> 🧐 I finished analyzing your image!\n```\n{caption}\n```", allowed_mentions=discord.AllowedMentions(users=True,replied_user=True))

    else:
        await interaction.response.send_message(
            "Please provide a valid image attachment."
        )







@tree.command(name="recharge", description="Recharge credits with Stripe")
async def recharge(
    interaction: discord.Interaction,
):
    user_id = interaction.user.id
    username = interaction.user.name

    # Make sure they exist in the db
    user_credits = await discord_balance_prompt(user_id, username)
    if user_credits is None:
        create_DB_user(user_id, username)

    payment_link = await create_payment_link(
        user_id, await get_default_pricing(stripe_product_id)
    )
    payment_link_url = payment_link.url

    if payment_link_url == "failed":
        await interaction.response.send_message(
            f"Failed to create payment link or payment itself failed. Please try again later.",
            ephemeral=True,
        )
        exit

    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO payments (user_id, type, timestamp, txid)
        VALUES (?, ?, ?, ?)
    """, (user_id, "stripe_payment_link", datetime.now(), payment_link.id))
    conn.commit()

    await interaction.response.send_message(
        f"Recharge your account: {payment_link.url}", ephemeral=True
    )
    await interaction.response.defer(ephemeral=True)


@tree.command(name="balance", description="Check your credit balance")
async def balance(
    interaction: discord.Interaction,
):
    user_id = interaction.user.id
    username = interaction.user.name
    # make sure they exist in the db
    user_credits = await discord_balance_prompt(user_id, username)
    if user_credits is None:
       await create_DB_user(user_id, username)

    await interaction.response.send_message(
        f"Your current balance is: {user_credits}", ephemeral=True
    )
    await interaction.response.defer(ephemeral=True)


def generate_bot_invite_link(client_id):
    base_url = "https://discord.com/api/oauth2/authorize"
    permissions = "8"  # Admin permissions
    scope = "bot"
    invite_link = (
        f"{base_url}?client_id={client_id}&permissions={permissions}&scope={scope}"
    )
    return invite_link


JOB_LOGGER = logging.getLogger("datapulse_jobs")

#@tasks.loop(seconds=15)
async def task_test():
    JOB_LOGGER.info("Stripe Events Check Started")
    start_time = time.perf_counter()
    await verify_payment_links_job()
    end_time = time.perf_counter()
    JOB_LOGGER.info(f"Stripe Events Check Done: {end_time - start_time} seconds")

# Example usage
client_id = "1222513177699422279"  # Replace with your bot's client ID
invite_link = generate_bot_invite_link(client_id)
print("Invite your bot using this link:", invite_link)

logging.getLogger('stripe').setLevel(logging.WARNING)

# run the bot
async def main():
    try:
        await client.start(TOKEN)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
