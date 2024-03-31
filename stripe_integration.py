import asyncio
import sqlite3
import stripe
from utils import config
from datetime import datetime

DATABASE_URL = "./config/database.sqlite"

async def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def create_customer(user_id, username, source):
    customer = await run_in_executor(
        stripe.Customer.create,
        metadata={
            'online_username': username,
            'username_from': source,
            'user_id': str(user_id),
        }
    )
    return customer.id

async def create_payment_link(user_id, price_id, customer_id=None):
    payment_link = await run_in_executor(
        stripe.PaymentLink.create,
        line_items=[{'price': price_id, 'quantity': 1}],
        metadata={'user_id': str(user_id), 'customer_id': customer_id or ''}
    )
    return payment_link.url

async def get_default_pricing(stripe_product_id):
    price = await run_in_executor(stripe.Price.retrieve, id=stripe_product_id)
    return price

def get_credit_per_usd():
    return config['credits']['stripe']['credits_per_dollar']

async def get_credit_amount_per_price_id(price_id):
    price_object = await run_in_executor(stripe.Price.retrieve, price_id)
    usd_amount = price_object.unit_amount / 100  # Stripe amounts are in cents
    return get_total_credit_amount(usd_amount)

def get_total_credit_amount(usd_amount):
    return usd_amount * get_credit_per_usd()

async def verify_payment_links_job():
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT * FROM payments
            WHERE confirmed_at IS NULL AND type = 'stripe_payment_link'
        """)
        unconfirmed_payments = cursor.fetchall()
        unconfirmed_links = [payment[4] for payment in unconfirmed_payments]  # Assuming txid is the 5th column in the payments table

        events = await run_in_executor(stripe.Event.list)
        for event in events.auto_paging_iter():
            if event['type'] != "checkout.session.completed":
                continue

            payment_link = event['data']['object'].get('payment_link')
            if payment_link and payment_link in unconfirmed_links:
                link = await run_in_executor(stripe.PaymentLink.retrieve, payment_link)
                user_id = link.metadata.get('user_id')
                total_usd = event['data']['object'].get('amount_total') / 100
                credits = get_total_credit_amount(total_usd)

                cursor.execute("""
                    SELECT * FROM users WHERE id = ?
                """, (user_id,))
                user = cursor.fetchone()

                payment_to_confirm = next((p for p in unconfirmed_payments if p[4] == payment_link), None)  # Assuming txid is the 5th column in the payments table
                if payment_to_confirm:
                    updated_credits = user[3] + credits  # Assuming credits is the 4th column in the users table
                    cursor.execute("""
                        UPDATE users SET credits = ? WHERE id = ?
                    """, (updated_credits, user_id))

                    confirmed_at = datetime.now()
                    cursor.execute("""
                        UPDATE payments SET confirmed_at = ? WHERE id = ?
                    """, (confirmed_at, payment_to_confirm[0]))  # Assuming id is the 1st column in the payments table

                    conn.commit()

                    discord_id = user[1]  # Assuming discord_id is the 2nd column in the users table
                    # Perform any further actions with the discord_id

    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        cursor.close()
        conn.close()

async def verify_payment_links():
    await verify_payment_links_job()