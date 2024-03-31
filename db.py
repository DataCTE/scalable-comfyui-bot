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
            UUID TEXT,
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

# Example usage
def main():
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        # Create a new user
        cursor.execute("""
            INSERT INTO users (username, email, user_id)
            VALUES (?, ?, ?)
        """, ("john_doe", "john@example.com", "123456"))
        conn.commit()
        logger.info("Created user: john_doe")

        # Create a credit entry for the user
        cursor.execute("""
            INSERT INTO credits (user_id, credits)
            VALUES (?, ?)
        """, ("123456", 200))
        conn.commit()
        logger.info("Created credit entry for user: 123456")

        # Retrieve user and credit information
        cursor.execute("""
            SELECT users.username, credits.credits
            FROM users
            JOIN credits ON users.user_id = credits.user_id
            WHERE users.user_id = ?
        """, ("123456",))
        result = cursor.fetchone()
        if result:
            username, credits = result
            logger.info(f"Retrieved user: {username}")
            logger.info(f"User credits: {credits}")
        else:
            logger.warning("User not found")

    except sqlite3.Error as e:
        logger.error(f"Database error: {str(e)}")

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()