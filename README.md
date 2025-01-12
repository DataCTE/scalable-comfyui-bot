# Scalable ComfyUI Bot

A distributed Discord bot for AI image generation with integrated credit system and payment processing. Supports load balancing across multiple ComfyUI instances.

> **Attribution**: This project builds upon the excellent foundation of [ComfyUI-SDXL-DiscordBot](https://github.com/aaronfisher-code/ComfyUI-SDXL-DiscordBot) by [Aaron Fisher](https://github.com/aaronfisher-code). The initial framework and ComfyUI integration were inspired by their work.

## Project History

This bot started as an internal tool for testing SDXL model iterations in a controlled environment. Key aspects of its development:

- **Private Testing**: Enabled testing of SDXL models without distributing weights
- **Feedback Collection**: Used to gather user feedback on model outputs during training
- **Credit System**: Implemented to manage and track usage during testing phases
- **Multiple Models**: Supported testing different model versions and architectures:
  - SDXL base models
  - Fine-tuned variations
  - Custom architectures (PixArt-900M, AuraFlow, Flux)

The bot evolved to include features like credit management, multiple model support, and image manipulation capabilities, making it a comprehensive platform for AI image generation testing.

> **Note**: The `/describe` command is currently under maintenance and may not function as expected.

## Project Structure
```
DATAPULSE_BOT/
├── src/
│   ├── comfyUI-workflows/      # ComfyUI workflow configurations
│   │   ├── describe_config.json
│   │   ├── gen_avatar_config.json
│   │   ├── img2img_config.json
│   │   ├── lora2img_config.json
│   │   ├── pixarttext2img_config.json
│   │   ├── style2img_config.json
│   │   ├── text2img_config.json
│   │   └── ... (other workflow configs)
│   │
│   ├── database/              # Database operations
│   │   ├── database_query_same_energy.py
│   │   ├── database_query.py
│   │   └── db.py
│   │
│   ├── services/             # Core services
│   │   ├── apiImageGen.py
│   │   ├── avatar_cog.py
│   │   ├── imageGen.py
│   │   ├── payment_service.py
│   │   ├── sdxl-comfyui-fastapi.py
│   │   └── stripe_integration.py
│   │
│   ├── utils/               # Utility functions
│   │   └── config.py
│   │
│   └── docs/                # Documentation
│       ├── LICENSE
│       └── README.md
│
├── config/                  # Configuration files
│   ├── payment.json
│   └── config.properties
│
├── notebooks/              # Jupyter notebooks
│   └── host.ipynb
│
├── logs/                   # Log files
│   └── logfile.txt
│
├── bot.py                  # Main bot application
├── requirements.txt        # Python dependencies
└── run.sh                 # Shell script to run the bot
```

## Features

1. **AI Image Generation**
   - Generate images from text prompts using the `/imagine` command
   - Multiple AI models supported:
     - stable-diffusion-xl
     - PixArt-900M
     - AuraFlow
     - Flux
     - Kolors
   - Style transfer and avatar generation capabilities
   - Support for negative prompts to refine generation

2. **Image Manipulation**
   - Upscale generated images for higher quality
   - Generate variations of existing images
   - Create image collages automatically
   - Style transfer between images

3. **Credit System**
   - Built-in credit management for users
   - Check balance with `/balance` command
   - Automatic credit deduction for image operations

4. **Payment Integration**
   - Seamless Stripe payment integration
   - Easy credit recharge with `/recharge` command
   - Secure payment processing and tracking

5. **Additional Features**
   - Image description capabilities with `/describe` command
   - Channel-specific command restrictions
   - User-friendly interactive buttons for image operations
   - Load balancing across multiple ComfyUI instances

## Setup

### Prerequisites
- Python 3.x
- Discord Bot Token
- Stripe API Key and Product ID
- Replicate API Token
- One or more ComfyUI instances

### Configuration
1. Create `config/config.properties` with the following structure:
```ini
[DISCORD]
TOKEN=your_discord_bot_token

[IMAGE]
SOURCE=LOCAL

[STRIPE]
API_KEY=your_stripe_api_key
PRODUCT_ID=your_stripe_product_id

[REPLICATE]
API_TOKEN=your_replicate_api_token

[LOCAL]
TYPE=cluster
SERVER_ADDRESS=127.0.0.1:8188

[COMFY_CLUSTER]
# Comma-separated list of ComfyUI instances
SERVER_ADDRESSES=127.0.0.1:8188,127.0.0.1:8189,127.0.0.1:8190
```

2. Set up ComfyUI workflows:
   - Place your workflow JSON files in `src/comfyUI-workflows/`
   - Required workflows:
     - `text2img_config.json`
     - `img2img_config.json`
     - `upscale_config.json`
     - `style2img_config.json`
     - And others as specified in config.properties

3. Database setup:
   - The bot automatically creates necessary tables on first run
   - Database file location: `config/database.sqlite`

### Running the Bot
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the bot:
```bash
./run.sh
```

## Usage

### Basic Commands
- `/imagine [prompt]` - Generate images from text
- `/balance` - Check your credit balance
- `/recharge` - Get a payment link to add credits
- `/describe [image]` - Get AI description of an image

### Image Generation Parameters
- `prompt` - Main text prompt for generation
- `negative_prompt` - What to avoid in generation
- `batch_size` - Number of images (max 4)
- `width` & `height` - Image dimensions
- `model` - AI model to use (stable-diffusion-xl, PixArt-900M, etc.)
- `cfg` - Configuration scale
- `steps` - Generation steps

### Clustering Setup
The bot supports distributed image generation across multiple ComfyUI instances:

1. Single Instance:
   ```ini
   [LOCAL]
   TYPE=single
   SERVER_ADDRESS=127.0.0.1:8188
   ```

2. Multiple Instances:
   ```ini
   [LOCAL]
   TYPE=cluster
   
   [COMFY_CLUSTER]
   SERVER_ADDRESSES=127.0.0.1:8188,127.0.0.1:8189,192.168.1.100:8188
   ```

The bot will automatically distribute requests across all configured instances in a round-robin fashion.

## Credits and Payments
- New users start with initial credits
- Each operation costs a specific amount of credits
- Recharge credits through Stripe payment system
- Secure payment processing and automatic credit allocation

## Security
- Channel-specific command restrictions
- DM commands disabled
- Secure payment processing through Stripe
- Database tracking of all transactions

## Support
For support or feature requests, please open an issue in the repository.
