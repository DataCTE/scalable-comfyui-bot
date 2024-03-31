import os
import configparser


def config():
    # Check for the existence of the config file and create default if it doesn't exist
    if not os.path.exists("config.properties"):
        generate_default_config()

    # Create the output directory if it doesn't exist
    if not os.path.exists("./out"):
        os.makedirs("./out")

    # Load the configuration
    config = configparser.ConfigParser()
    config.read("config.properties")
    
    # Extract necessary configuration values
    discord_token = config["DISCORD"]["TOKEN"]
    stripe_api_key = config["STRIPE"]["API_KEY"]
    stripe_product_id = config["STRIPE"]["PRODUCT_ID"]
    
    # Return the extracted values
    return {
        "discord_token": discord_token,
        "stripe_api_key": stripe_api_key,
        "stripe_product_id": stripe_product_id
    }

def generate_default_config():
    config = configparser.ConfigParser()
    config["DISCORD"] = {"TOKEN": ""}
    config["LOCAL"] = {"SERVER_ADDRESS": "http://127.0.0.1:9191"}
    config["API"] = {
        "API_KEY": "STABILITY_AI_API_KEY",
        "API_HOST": "https://api.stability.ai",
        "API_IMAGE_ENGINE": "STABILITY_AI_IMAGE_GEN_MODEL",
    }
    with open("config.properties", "w") as configfile:
        config.write(configfile)


