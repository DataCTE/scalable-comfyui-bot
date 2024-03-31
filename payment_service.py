import sqlite3
from datetime import datetime
from stripe_integration import create_payment_link, get_default_pricing, verify_payment_links_job

DATABASE_URL = "./config/database.sqlite"

async def create_DB_user(user_id, username):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO users (user_id, username, credits)
            VALUES (?, ?, ?)
        """, (user_id, username, 100))
        conn.commit()
        return True

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return False

    finally:
        cursor.close()
        conn.close()

async def deduct_credits(user_id, amount):
    """Deducts a specified amount of credits from the user's account."""
    user_id = str(user_id)

    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM users WHERE user_id = ?
        """, (user_id,))
        user = cursor.fetchone()

        if user:
            current_credits = user[3]  # Assuming credits is the 4th column in the users table
            new_credits = current_credits - amount

            cursor.execute("""
                UPDATE users SET credits = ? WHERE user_id = ?
            """, (new_credits, user_id))
            conn.commit()
            print(f"Credits deducted successfully for user {user_id}")
            return True
        else:
            print(f"User {user_id} not found")
            return False

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return False

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

    finally:
        cursor.close()
        conn.close()

async def ensure_stripe_customer_exists(user_id, username, source):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM users WHERE user_id = ?
        """, (user_id,))
        user = cursor.fetchone()

        if user is None:
            cursor.execute("""
                INSERT INTO users (user_id, username, source)
                VALUES (?, ?, ?)
            """, (user_id, username, source))
            conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")

    finally:
        cursor.close()
        conn.close()

async def discord_recharge_prompt(username, user_id):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        source = "discord"
        cursor.execute("""
            SELECT * FROM users WHERE user_id = ?
        """, (user_id,))
        user = cursor.fetchone()

        if user is None:
            source = "other"
            return source

        await ensure_stripe_customer_exists(user_id=user_id, username=username, source=source)
        payment_link = await create_payment_link(user_id, get_default_pricing())

        if payment_link:
            cursor.execute("""
                INSERT INTO payments (user_id, type, timestamp, txid)
                VALUES (?, ?, ?, ?)
            """, (user_id, "stripe_payment_link", datetime.now(), payment_link.id))
            conn.commit()
            return payment_link.url
        else:
            return "failed"

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return "failed"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "failed"

    finally:
        cursor.close()
        conn.close()

async def discord_balance_prompt(user_id, username):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM users WHERE user_id = ?
        """, (user_id,))
        user = cursor.fetchone()
        print(user)

        if user is None:
            await create_DB_user(user_id, username)
            cursor.execute("""
                SELECT * FROM users WHERE user_id = ?
            """, (user_id,))
            user = cursor.fetchone()

        if user is not None:
            credits = user[3]  # Assuming credits is the 4th column in the users table
            print(user, credits, user_id, username)
            return credits
        else:
            print(user, user_id, username)
            return None

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        cursor.close()
        conn.close()