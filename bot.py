import discord
import discord.ext
from discord import app_commands
from discord.app_commands import Choice
import configparser
from PIL import Image
from datetime import datetime
from db import init_db
from imageGen import *
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


# setting up the bot
config = configparser.ConfigParser()
config.read("config.properties")
TOKEN = config["DISCORD"]["TOKEN"]
IMAGE_SOURCE = config["IMAGE"]["SOURCE"]
stripe_api_key = config["STRIPE"]["API_KEY"]
stripe_product_id = config.get("STRIPE", "PRODUCT_ID")


stripe.api_key = stripe_api_key
intents = discord.Intents.all()
intents.members = True  # Enable the members intent
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)


if IMAGE_SOURCE == "LOCAL":
    server_address = config.get("LOCAL", "SERVER_ADDRESS")
    from imageGen import generate_images, upscale_image, generate_alternatives, lora_images






model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


async def generate_caption(image):
    # Convert the image to RGB format
    image = image.convert("RGB")

    # Process the image and generate the caption
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)  # Move the pixel values to the GPU

    output = model.generate(pixel_values, max_length=50, num_beams=4)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)

    return caption


async def extract_index_from_id(custom_id):
    try:
        # Extracting numeric part while assuming it might contain non-numeric characters
        numeric_part = "".join([char for char in custom_id[1:] if char.isdigit()])
        if not numeric_part:
            return None
        return int(numeric_part) - 1
    except ValueError:
        return None


class ImageButton(discord.ui.Button):
    def __init__(self, label, emoji, row, callback):
        super().__init__(
            label=label, style=discord.ButtonStyle.grey, emoji=emoji, row=row
        )
        self._callback = callback

    async def callback(self, interaction: discord.Interaction):
        await self._callback(interaction, self)


class Buttons(discord.ui.View):
    def __init__(
        self,
        prompt,
        negative_prompt,
        UUID,
        user_id,
        url,
        model,
        images,
        timeout=10000000000000,
    ):
        super().__init__(timeout=timeout)
        self.UUID = UUID  # Store the UUID
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.user_id = user_id
        self.url = url
        self.model = model
        self.images = images

        total_buttons = (
            len(self.images) * 2 + 1
        )  # For both alternative and upscale buttons + re-roll button
        if total_buttons > 25:  # Limit to 25 buttons
            self.images = self.images[:12]  # Adjust to only use the first 12 images

        # Determine if re-roll button should be on its own row
        reroll_row = 1 if total_buttons <= 21 else 0

        # Inside the Buttons class __init__ method, adjust the button callbacks setup
        for idx, image in enumerate(self.images):
            row = (idx + 1) // 5 + reroll_row
            # Use functools.partial to correctly prepare the callback with necessary arguments
            reroll_callback = functools.partial(self.reroll_image, count=idx + 1)
            btn = ImageButton(f"V{idx + 1}", "♻️", row, reroll_callback)
            self.add_item(btn)

        for idx, image in enumerate(self.images):
            row = (idx + len(self.images) + 1) // 5 + reroll_row
            # Similarly adjust for upscale_image
            upscale_callback = functools.partial(self.upscale_image, count=idx + 1)
            btn = ImageButton(f"U{idx + 1}", "⬆️", row, upscale_callback)
            self.add_item(btn)

    @classmethod
    async def create(cls, prompt, negative_prompt, UUID, user_id, url, model):
        def get_images():
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    SELECT * FROM images
                    WHERE UUID = ? AND user_id = ?
                    ORDER BY count
                """,
                    (UUID, user_id),
                )
                images = cursor.fetchall()
                return images

            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")
                return []

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return []

            finally:
                cursor.close()
                conn.close()

        images = await asyncio.to_thread(get_images)

        return cls(prompt, negative_prompt, UUID, user_id, url, model, images)

    async def reroll_image(
        self, interaction: discord.Interaction, button: discord.ui.Button, count: int
    ):
        try:
            batch_size = 4
            # Grab the button number and then convert that to count to grab with the UUID from the db
            index = await extract_index_from_id(button.custom_id)
            if index is None:
                await interaction.response.send_message(
                    "Invalid custom_id format. Please ensure it contains a numeric index."
                )
                return

            await interaction.response.send_message(
                f"{interaction.user.mention} asked me to re-imagine the image, this shouldn't take too long..."
            )
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
                cursor.execute(
                    """
                    SELECT prompt FROM images
                    WHERE UUID = ? AND COUNT = ? LIMIT 1
                """,
                    (self.UUID, count),
                )
                result = cursor.fetchone()
                prompt = (
                    result[0] if result else self.prompt
                )  # Use the original prompt as a fallback

                # Generate a new UUID for the re-rolled image
                new_UUID = str(uuid.uuid4())

                # Generate a new image with the retrieved prompt
                await generate_alternatives(
                    UUID=new_UUID,
                    index=index,
                    user_id=interaction.user.id,
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    batch_size=batch_size,
                    width=1024,
                    height=1024,
                    model=self.model,
                )

                # Create a new collage for the re-rolled image
                collage_path = await create_collage(UUID=new_UUID)

                # Construct the final message with user mention
                final_message = f'{interaction.user.mention} asked me to re-imagine the image, here is what I imagined for them. "{prompt}", "{self.model}"'
                await interaction.channel.send(
                    content=final_message,
                    file=discord.File(fp=collage_path, filename="collage.png"),
                    view=Buttons(
                        prompt,
                        self.negative_prompt,
                        new_UUID,
                        interaction.user.id,
                        collage_path,
                        self.model,
                    ),
                )
                # after successful reroll, deduct credits
                amount = user_credits - 10
                await deduct_credits(user_id, amount)

            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")

            except Exception as e:
                print(f"An error occurred: {str(e)}")

            finally:
                cursor.close()
                conn.close()

        except discord.errors.InteractionResponded:
            return print("interaction failed")

    async def upscale_image(
        self, interaction: discord.Interaction, button: discord.ui.Button, count: int
    ):
        try:
            index = await extract_index_from_id(button.custom_id)
            if index is None:
                await interaction.response.send_message(
                    "Invalid custom_id format. Please ensure it contains a numeric index."
                )
                return

            await interaction.response.send_message(
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
                cursor.execute(
                    """
                    SELECT prompt FROM images
                    WHERE UUID = ? AND COUNT = ? LIMIT 1
                """,
                    (self.UUID,),
                )
                images = cursor.fetchall()

                if index < len(images):
                    image = images[index]
                    image_data = image[
                        2
                    ]  # Assuming image data is stored in the 3rd column

                    # Upscale image logic assumed to be defined elsewhere
                    upscaled_image = await upscale_image(
                        image_data, self.prompt, self.negative_prompt
                    )

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    upscaled_image_path = f"./out/upscaledImage_{timestamp}.png"

                    # Assuming upscaled_image has a .save() method
                    upscaled_image.save(upscaled_image_path)

                    final_message = (
                        f"{interaction.user.mention} here is your upscaled image"
                    )
                    await interaction.channel.send(
                        content=final_message,
                        file=discord.File(
                            fp=upscaled_image_path, filename="upscaled_image.png"
                        ),
                    )
                    # deduct credits
                    amount = user_credits - 5
                    print(amount)
                    await deduct_credits(user_id, amount)
                else:
                    await interaction.followup.send("Invalid image index.")

            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")

            except Exception as e:
                print(f"An error occurred: {str(e)}")

            finally:
                cursor.close()
                conn.close()

        except discord.errors.InteractionResponded:
            print("Interaction already responded")





@tree.command(name="describe", description="Describe an image")
@app_commands.describe(image="Image to describe")
async def describe(interaction: discord.Interaction, image: discord.Attachment):
    if image:
        # Retrieve the image data from the attachment
        image_data = await image.read()

        # Open the image using Pillow
        image = Image.open(BytesIO(image_data))

        # Generate the caption using the pre-trained model
        caption = await asyncio.to_thread(generate_caption, image)

        # Send the generated caption as a response
        await interaction.response.send_message(content=caption)
    else:
        await interaction.response.send_message(
            "Please provide a valid image attachment."
        )



# Assuming 'bot' is your commands.Bot or app_commands.CommandTree instance
@tree.command(name="imagine", description="Generate an image based on input text")
@app_commands.describe(prompt="Prompt for the image being generated")
@app_commands.describe(negative_prompt="Prompt for what you want to steer the AI away from")
@app_commands.describe(batch_size="Number of images to generate" )
@app_commands.describe(width="width of the image")
@app_commands.describe(height="height of the image" )
@app_commands.describe(attachment="attachment to use")
@app_commands.choices(model=[
    Choice(name="ProteusV1", value="proteus-rundiffusionV2.5"),
    Choice(name="Anime", value="AnimeP")
])
@app_commands.describe(lora="Choose the lora to use")
@app_commands.choices(lora=[
    Choice(name="Detail", value="tweak-detail-xl"),
    Choice(name="SythAnimeV2", value="AnimeSythenticV0.2"),
    Choice(name="Artistic", value="XL_more_art-full_v1")
])
async def imagine(
    interaction: discord.Interaction, 
    prompt: str, 
    negative_prompt: str = None,
    batch_size: int = 4,
    width: int = 1024,
    height: int = 1024,
    model: str = "proteus-rundiffusionV2.5",
    attachment: discord.Attachment = None, 
    lora: str = None
):
    user_id = interaction.user.id
    username = interaction.user.name
    user_credits = await discord_balance_prompt(user_id, username)

    if user_credits is None:
        # Handle case where user credits couldn't be retrieved
        create_DB_user(user_id, username) 

    elif user_credits < 10:  # Assuming 5 credits are needed
        payment_link = await discord_recharge_prompt(username, user_id)
        if payment_link == "failed":
            await interaction.response.send_message(
                "Failed to create payment link or payment itself failed. Please try again later.",
                ephemeral=True
            )
           
        await interaction.response.send_message(
            f"You don't have enough credits. Please recharge your account: {payment_link}",
            ephemeral=True
        )
        await interaction.response.defer(ephemeral=True)
    else:
        await interaction.response.defer(ephemeral=False)



    

    UUID = str(uuid.uuid4())  # Generate unique hash for each image
    prompt_gen = f"{prompt}, masterpiece, best quality"

    if attachment:
        # save input image

        await style_images(
            UUID=UUID,
            user_id=interaction.user.id,
            prompt=prompt_gen,
            attachment=attachment,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            model=model,
        )
    if lora is not None:
        await lora_images(
            UUID=UUID,
            user_id=interaction.user.id,
            prompt=prompt_gen,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            model=model,
            lora=lora,
        )
    else:
        await generate_images(
            UUID=UUID,
            user_id=interaction.user.id,
            prompt=prompt_gen,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            model=model,
            lora=lora,
        )

    collage_path = await create_collage(UUID)

    if collage_path is None:
        print(collage_path)
        return

    buttons_view = await Buttons.create(
        prompt,
        negative_prompt,
        UUID,
        interaction.user.id,
        collage_path,
        model,
        batch_size,
    )

    file = discord.File(collage_path, filename="collage.png")
    final_message = f'{interaction.user.mention}, here is what I imagined for you with "{prompt}", "{model}":'
    
    await interaction.followup.send(content=final_message, file=file, view=buttons_view, ephemeral=False)
    if user_id == "879714655356997692":
        print("User ID matches the specified value. Skipping credit deduction.")
    else:
        amount = user_credits - 10
        print(amount)
        await deduct_credits(user_id, amount)

    return UUID


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
    if payment_link == "failed":
        await interaction.response.send_message(
            f"Failed to create payment link or payment itself failed. Please try again later.",
            ephemeral=True,
        )
        exit
    await interaction.response.send_message(
        f"Recharge your account: {payment_link}", ephemeral=True
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
        create_DB_user(user_id, username)

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


client.run(TOKEN)
