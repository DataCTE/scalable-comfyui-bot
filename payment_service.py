import sqlite3
from datetime import datetime
from stripe_integration import create_payment_link, get_default_pricing, verify_payment_links_job

DATABASE_URL = "./config/database.sqlite"

async def create_DB_user(user_id:str, username: str):
    """
    Creates a new user in the database with the specified user_id and username.
    
    Parameters:
    user_id (str): The user's ID to be created.
    username (str): The user's username to be created.
    ----------
    Returns:
    bool: True if the user was created successfully, False otherwise.
    
    --------
    Example:
    >>> create_DB_user("1234567890", "JohnDoe")
    True
    # User 1234567890 created successfully with the username JohnDoe

    """
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

async def deduct_credits(user_id:str, amount:int):
    """Deducts a specified amount of credits from the user's account.
    
    Parameters:
    user_id (str): The user's ID whose credits are to be deducted.
    amount (int): The amount of credits to be deducted.
    ----------
    Returns:
    bool: True if the credits were deducted successfully, False otherwise.
    
    --------
    Example:
    >>> deduct_credits("1234567890", 50)
    True
    # Credits deducted successfully for user 1234567890 in the amount of 50

    """
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

async def ensure_stripe_customer_exists(user_id:str, username:str, source:str):
    """Ensures that a Stripe customer exists for the specified user.
    
    Parameters:
    user_id (str): The user's ID whose Stripe customer is to be ensured.
    username (str): The user's username whose Stripe customer is to be ensured.
    source (str): The source of the user.
    ----------
    Returns:
    None
    
    --------
    Example:
    >>> ensure_stripe_customer_exists("1234567890", "JohnDoe", "discord")
    None
    # Stripe customer ensured for user 1234567890 with the username JohnDoe
    """
    
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

async def discord_recharge_prompt(user_id:str, username:str):
    """Prompts the user to recharge their account. If the prompt is successful, a payment link is generated.

    Parameters:
    user_id (str): The user's ID to be prompted.
    username (str): The user's username to be prompted.
    ----------
    Returns:
    str: The payment link URL if the prompt was successful, "failed" otherwise.
    
    --------
    Example:
    >>> discord_recharge_prompt("1234567890", "JohnDoe")
    "https://example.com"
    # Payment link generated successfully for user 1234567890 with the username JohnDoe
    """
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

async def discord_balance_prompt(user_id:str, username:str):
    """Prompts the user to check their account balance.

    Parameters:
    user_id (str): The user's ID to be prompted.
    username (str): The user's username to be prompted.
    ----------
    Returns:
    str: The user's account balance if the prompt was successful, "failed" otherwise.

    --------
    Example:
    >>> discord_balance_prompt("1234567890", "JohnDoe")
    "100"
    # Account balance checked successfully for user 1234567890 with the username JohnDoe
    """
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