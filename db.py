import logging
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = "./config/database.sqlite"

def create_tables(conn):
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            user_id TEXT UNIQUE,
            stripeID TEXT,
            hiveID TEXT,
            tier INTEGER DEFAULT 0,
            banned BOOLEAN DEFAULT FALSE
        )
    """)

    # Create credits table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS credits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            credits INTEGER DEFAULT 100,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    # Create guilds table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guilds (
            id INTEGER PRIMARY KEY,
            name TEXT,
            bannedguild BOOLEAN DEFAULT FALSE,
            owner INTEGER
        )
    """)

    # Create channels table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            id INTEGER PRIMARY KEY,
            nsfw BOOLEAN DEFAULT FALSE,
            name TEXT
        )
    """)

    # Create payments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            type TEXT,
            timestamp DATETIME,
            txid TEXT,
            confirmedAt DATETIME,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    # Create images table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            data BLOB,
            UUID_list TEXT,
            url TEXT,
            count INTEGER,
            model TEXT,
            prompt TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    # Create jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data JSON
        )
    """)

    conn.commit()
    logger.info("Database tables created")

def init_db():
    db_dir = os.path.dirname(DATABASE_URL)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        logger.info(f"Created directory: {db_dir}")

    conn = sqlite3.connect(DATABASE_URL)
    create_tables(conn)
    conn.close()

async def save_image_generation(user_id: str, prompt: str, image_path: str):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO images (user_id, prompt, url)
            VALUES (?, ?, ?)
        """, (user_id, prompt, image_path))
        conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        cursor.close()
        conn.close()

async def get_user_images(user_id: str):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM images WHERE user_id = ?
        """, (user_id,))
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

