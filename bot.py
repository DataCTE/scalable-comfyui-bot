import discord
import discord.ext
from discord import app_commands
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

# setting up the bot
config = configparser.ConfigParser()
config.read("config.properties")
TOKEN = config["DISCORD"]["TOKEN"]
IMAGE_SOURCE = config["IMAGE"]["SOURCE"]
stripe_api_key=config["STRIPE"]["API_KEY"]  
stripe_product_id = config.get("STRIPE", "PRODUCT_ID")

# Initialize the bot and

stripe.api_key = stripe_api_key
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

if IMAGE_SOURCE == "LOCAL":
    server_address = config.get("LOCAL", "SERVER_ADDRESS")
    from imageGen import generate_images, upscale_image, generate_alternatives
elif IMAGE_SOURCE == "API":
    config.get["API", "API_KEY"]["API", "API_HOST"]
    from apiImageGen import generate_images, upscale_image, generate_alternatives


# sync the slash command to your server
@client.event
async def on_ready():
    init_db()  # Initialize DB
    await tree.sync()
    print(f"Logged in as {client.user.name} ({client.user.id})")


async def extract_index_from_id(custom_id):
    try:
        # Extracting numeric part while assuming it might contain non-numeric characters
        numeric_part = ''.join([char for char in custom_id[1:] if char.isdigit()])
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
    def __init__(self, prompt, negative_prompt, UUID, user_id, url, model, images, timeout=10000000000000):
        super().__init__(timeout=timeout)
        self.UUID = UUID  # Store the UUID
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.user_id = user_id
        self.url = url
        self.model = model
        self.images = images
    


        total_buttons = len(self.images) * 2 + 1  # For both alternative and upscale buttons + re-roll button
        if total_buttons > 25:  # Limit to 25 buttons
            self.images = self.images[:12]  # Adjust to only use the first 12 images

        # Determine if re-roll button should be on its own row
        reroll_row = 1 if total_buttons <= 21 else 0

        # Dynamically add alternative buttons
        for idx, image in enumerate(self.images):
            row = (idx + 1) // 5 + reroll_row  # Determine row based on index and re-roll row
            self.reroll_image = self.reroll_image(count=idx + 1)
            btn = ImageButton(f"V{idx + 1}", "♻️", row, self.reroll_image)
            self.add_item(btn)

        # Dynamically add upscale buttons
        for idx, image in enumerate(self.images):
            row = (idx + len(self.images) + 1) // 5 + reroll_row  # Determine row based on index, number of alternative buttons, and re-roll row
            self.reroll_image = self.reroll_image(count=idx + 1)
            btn = ImageButton(f"U{idx + 1}", "⬆️", row, self.upscale_image)
            self.add_item(btn)

    @classmethod
    async def create(cls, prompt, negative_prompt, UUID, user_id, url, model):
        def get_images():
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    SELECT * FROM images
                    WHERE UUID = ? AND user_id = ?
                    ORDER BY count
                """, (UUID, user_id))
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

    
    async def reroll_image(self, interaction: discord.Interaction, button: discord.ui.Button, count: int):
        try:
            batch_size = 4
            # Grab the button number and then convert that to count to grab with the UUID from the db
            index = await extract_index_from_id(button.custom_id)
            if index is None:
                await interaction.response.send_message("Invalid custom_id format. Please ensure it contains a numeric index.")
                return

            await interaction.response.send_message(
                f'{interaction.user.mention} asked me to re-imagine the image, this shouldn\'t take too long...'
            )
            button.disabled = True
            await interaction.message.edit(view=self)

            # Retrieve the prompt from the database
            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    SELECT prompt FROM images
                    WHERE UUID = ? AND COUNT = ? LIMIT 1
                """, (self.UUID, count))
                result = cursor.fetchone()
                prompt = result[0] if result else self.prompt  # Use the original prompt as a fallback

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
                    image=self.url
                )

                # Create a new collage for the re-rolled image
                collage_path = await create_collage(UUID=new_UUID)

                # Construct the final message with user mention
                final_message = f'{interaction.user.mention} asked me to re-imagine the image, here is what I imagined for them. "{prompt}", "{self.model}"'
                await interaction.channel.send(
                    content=final_message,
                    file=discord.File(fp=collage_path, filename="collage.png"),
                    view=Buttons(prompt, self.negative_prompt, new_UUID, interaction.user.id, collage_path, self.model)
                )

            except sqlite3.Error as e:
                print(f"Database error: {str(e)}")

            except Exception as e:
                print(f"An error occurred: {str(e)}")

            finally:
                cursor.close()
                conn.close()

        except discord.errors.InteractionResponded:
            return print("interaction failed")
        
    
    async def upscale_image(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            index = await extract_index_from_id(button.custom_id)
            if index is None:
                await interaction.response.send_message("Invalid custom_id format. Please ensure it contains a numeric index.")
                return

            await interaction.response.send_message(
                f'{interaction.user.mention} asked me to upscale the image, this shouldn\'t take too long...'
            )
            button.disabled = True
            await interaction.message.edit(view=self)

            conn = sqlite3.connect(DATABASE_URL)
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    SELECT * FROM images
                    WHERE UUID = ?
                    ORDER BY count
                """, (self.UUID,))
                images = cursor.fetchall()

                if index < len(images):
                    image = images[index]
                    image_data = image[2]  # Assuming image data is stored in the 3rd column

                    # Upscale image logic assumed to be defined elsewhere
                    upscaled_image = await upscale_image(image_data, self.prompt, self.negative_prompt)

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    upscaled_image_path = f"./out/upscaledImage_{timestamp}.png"

                    # Assuming upscaled_image has a .save() method
                    upscaled_image.save(upscaled_image_path)

                    final_message = f"{interaction.user.mention} here is your upscaled image"
                    await interaction.channel.send(
                        content=final_message,
                        file=discord.File(fp=upscaled_image_path, filename="upscaled_image.png")
                    )
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
        




@tree.command(name="imagine", description="Generate an image based on input text")
@app_commands.describe(prompt="Prompt for the image being generated")
@app_commands.describe(negative_prompt="Prompt for what you want to steer the AI away from")
@app_commands.describe(batch_size="Number of images to generate" )
@app_commands.describe(width="width of the image")
@app_commands.describe(height="height of the image" )
@app_commands.describe(model="model to use")
@app_commands.describe(attachment="attachment to use")
async def imagine(
    interaction: discord.Interaction,
    prompt: str,
    negative_prompt: Optional[str] = None,
    batch_size: int = 4,
    width: int = 1024,
    height: int = 1024,
    model: str = "proteus-rundiffusionV2.5",
    attachment: Optional[discord.Attachment] = None,
):
    username = interaction.user.name
    user_id = interaction.user.id

    user_credits = await discord_balance_prompt(user_id=user_id, username=username)

    if user_credits is None:
        # Handle case where user credits couldn't be retrieved
        # This could happen due to an error or if the user doesn't exist
        # You might want to log this or handle it based on your application logic
        await interaction.response.send_message(
            print(user_credits, user_id, username),
            "Failed to retrieve user credits. Please try again later.",
            ephemeral=True
        )
        await interaction.response.defer(ephemeral=True)
    elif user_credits < 5:  # Assuming 5 credits are needed
        payment_link = await discord_recharge_prompt(username, user_id)
        if payment_link == "failed":
            await interaction.response.send_message(
                "Failed to create payment link or payment itself failed. Please try again later.",
                ephemeral=True
            )
            exit  # Make sure to fix this to properly exit the function
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
        await style_images(
            UUID=UUID,
            user_id=interaction.user.id,
            prompt=prompt_gen,
            attachment=attachment.url if attachment else None,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            width=width,
            height=height,
            model=model,
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
        )

    collage_path = await create_collage(UUID)
    buttons_view = await Buttons.create(prompt, negative_prompt, UUID, interaction.user.id, collage_path, model)

    file = discord.File(collage_path, filename="collage.png")
    final_message = f'{interaction.user.mention}, here is what I imagined for you with "{prompt}", "{model}":'
    
    await interaction.followup.send(content=final_message, file=file, view=buttons_view, ephemeral=False)

    await deduct_credits(user_id, 5)
    
    
    return UUID 

@tree.command(name="recharge", description="Recharge credits with Stripe")
async def recharge(
    interaction: discord.Interaction,

):
    user_id = interaction.user.id
    payment_link = await create_payment_link(user_id, await get_default_pricing(stripe_product_id))
    if payment_link == "failed":
        await interaction.response.send_message(
            f"Failed to create payment link or payment itself failed. Please try again later.",
            ephemeral=True
        )
        exit
    await interaction.response.send_message(
        f"Recharge your account: {payment_link}",
        ephemeral=True
    )
    await interaction.response.defer(ephemeral=True)

@tree.command(name="balance", description="Check your credit balance")
async def balance(
    interaction: discord.Interaction,
):
    user_id = interaction.user.id
    username = interaction.user.name
    user_credits = await discord_balance_prompt(user_id=user_id,  username=username)
    await interaction.response.send_message(
        f"Your current balance is: {user_credits}",
        ephemeral=True
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


# Example usage
client_id = "1222513177699422279"  # Replace with your bot's client ID
invite_link = generate_bot_invite_link(client_id)
print("Invite your bot using this link:", invite_link)

# run the bot
client.run(TOKEN)
