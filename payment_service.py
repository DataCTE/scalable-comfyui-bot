# Import necessary modules
import asyncio
from db import User, Payment  # Assuming db.py contains the converted definitions
from stripe_integration import (
    StripeIntegration
)  # Assuming stripe_integration.py exists
from utils import config
import datetime as Date
from db import AsyncSessionLocal
from stripe_integration import StripeIntegration
async def fetch_user_by_discord(username, user_id):
    async with AsyncSessionLocal() as session:
        user = await session.get(User, discordID=user_id)
        if user is None:
            # When creating a new user, set the credits to 100
            user = User(username=username, discordID=user_id, credits=100)
            session.add(user)
            await session.commit()
            is_created = True
        else:
            is_created = False
    return user, is_created

async def deduct_credits(user):
    """Deducts a specified amount of credits from the user's account."""
    # Deduct flat 5 credits for each transaction
    amount = 5  # Deduct 5 credits
    if user is None:
        return False  # User not found, cannot deduct credits

    # Deduct credits from user's account
    user.credits -= amount

    # Save changes to the database
    async with AsyncSessionLocal() as session:
        session.add(user)
        await session.commit()

    return True

async def ensure_stripe_customer_exists(user, source):
    async with AsyncSessionLocal() as session:
        if not user.stripeID:
            customer_id = await StripeIntegration.create_customer(user.username, source)
            user.stripeID = customer_id
            session.add(user)
            await session.commit()
            return customer_id
        else:
            return user.stripeID

async def discord_recharge_prompt(username, user_id):
    async with AsyncSessionLocal() as session:
        user = await session.get(User, discordID=user_id)
        if user is None:
            user = User(username=username, discordID=user_id, credits=100)
            session.add(user)
            await session.commit()

        stripe_customer_id = await ensure_stripe_customer_exists(user, "discord")
        payment_link = await StripeIntegration.create_payment_link(
            user.id, StripeIntegration.get_default_pricing(), stripe_customer_id
        )
        if payment_link:
            # Create a pending payment record in the database
            payment = Payment(user_id=user.id, type="stripe_payment_link", timestamp=Date.now(), txid=payment_link.id)
            session.add(payment)
            await session.commit()
            return payment_link.url
        else:
            return "failed"
