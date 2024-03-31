from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import *
from sqlalchemy.dialects.sqlite import BLOB
import os
from sqlalchemy.ext.asyncio import AsyncEngine

DATABASE_URL = "sqlite+aiosqlite:///./config/database.sqlite"

# Create the asynchronous engine and session maker
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)

# Declare the base class for models
Base = declarative_base()

def deduct_credits(self, session, amount):
        """Deducts a specified amount of credits from the user's account."""
        if self.credits >= amount:
            self.credits -= amount
            session.commit()
            return True
        return False

# Define your models
class User(Base):  # Assuming User is a subclass of Base
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    user_id = Column(String, unique=True, index=True)
    stripeID = Column(String, index=True)
    hiveID = Column(String, index=True)
    credits = Column(Integer, default=100)
    tier = Column(Integer, default=0)
    banned = Column(Boolean, default=False)

    def __init__(self, username, user_id, credits):
        super(User, self).__init__(username=username, user_id=user_id, credits=credits)
        self.credits = credits


class Guild(Base):
    __tablename__ = "guilds"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    banned = Column(Boolean, default=False)
    owner = Column(Integer)


class Channel(Base):
    __tablename__ = "channels"
    id = Column(Integer, primary_key=True, index=True)
    nsfw = Column(Boolean, default=False)
    name = Column(String)


class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(String)
    timestamp = Column(DateTime)
    txid = Column(String)
    confirmedAt = Column(DateTime)


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    data = Column(BINARY)
    UUID = Column(String)
    url = Column(String)
    count = Column(Integer, default=None)
    model = Column(String, default=None)
    prompt = Column(String)


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    data = Column(JSON)


async def init_db(engine: AsyncEngine):
    # Extract the file path from the DATABASE_URL
    db_file_path = DATABASE_URL.split("///")[
        -1
    ]  # Get the path after 'sqlite+aiosqlite:///'
    db_dir = os.path.dirname(db_file_path)

    # Ensure the directory for the SQLite database exists
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Now proceed to create the tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Make sure to import and call init_db() at the appropriate place in your application
# Typically, this would be done at application startup
