from sqlalchemy.future import select
from sqlalchemy import update
from src.database.db import User, AsyncSessionLocal

async def get_user_balance(user_id: int) -> float:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User.credits).where(User.id == user_id))
        balance = result.scalar_one_or_none()
        return float(balance) if balance is not None else 0.0

async def decrement_credits(user_id: int, amount: float = 1.0) -> bool:
    async with AsyncSessionLocal() as session:
        await session.execute(update(User).where(User.id == user_id).values(credits=User.credits - amount))
        await session.commit()
    return True

async def increment_credits(user_id: int, amount: float = 1.0) -> bool:
    async with AsyncSessionLocal() as session:
        await session.execute(update(User).where(User.id == user_id).values(credits=User.credits + amount))
        await session.commit()
    return True

async def transfer_credits(from_user_id: int, to_user_id: int, amount: float = 1.0) -> bool:
    async with AsyncSessionLocal() as session:
        # Decrement from the sender
        await session.execute(update(User).where(User.id == from_user_id).values(credits=User.credits - amount))
        # Increment for the receiver
        await session.execute(update(User).where(User.id == to_user_id).values(credits=User.credits + amount))
        await session.commit()
    return True
