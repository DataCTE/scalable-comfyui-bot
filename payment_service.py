# Import necessary modules
import asyncio
from db import User, Payment  # Assuming db.py contains the converted definitions

from utils import config
import datetime as Date
from db import AsyncSessionLocal
from stripe_integration import *
from sqlalchemy.future import select

from sqlalchemy.exc import SQLAlchemyError


async def deduct_credits(user_id, amount):
    """Deducts a specified amount of credits from the user's account."""
    user_id = str(user_id)

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.user_id == user_id))
            user = result.scalar_one_or_none()

            if user:
                user.credits -= amount

                session.add(user)
                await session.commit()
                print(f"Credits deducted successfully for user {user_id}")
                return True
            else:
                print(f"User {user_id} not found")
                return False

    except SQLAlchemyError as e:
        print(f"Database error: {str(e)}")
        return False

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

async def ensure_stripe_customer_exists(user_id, username, source):
    async with AsyncSessionLocal() as session:
        # Check if the user already exists
        result = await session.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()

        if user is None:
            # The user does not exist, create a new one
            user = User(user_id=user_id, username=username, source=source)
            session.add(user)
            await session.commit()
    

async def discord_recharge_prompt(username, user_id):
    async with AsyncSessionLocal() as session:
        source="discord"
        # Fetch the user from the database
        result = await session.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if user_id is None:
            source="other"
            return source
        await ensure_stripe_customer_exists(user_id=user_id, username=username, source=source)
        payment_link = await create_payment_link(user_id, get_default_pricing())
        if payment_link:
            # Create a pending payment record in the database
            payment = Payment(user_id=user.id, type="stripe_payment_link", timestamp=Date.now(), txid=payment_link.id)
            session.add(payment)
            await session.commit()
            return payment_link.url
        else:
            return "failed"