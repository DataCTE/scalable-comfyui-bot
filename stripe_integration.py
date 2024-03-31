import asyncio
import stripe
from db import Payment, User  # Ensure these models exist and are correctly defined
from utils import config  # Ensure this contains your configuration details
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from db import AsyncSessionLocal
from sqlalchemy.future import select
class StripeIntegration:
    def __init__(self, stripe_api_key):
        stripe.api_key = stripe_api_key
        self.executor = ThreadPoolExecutor(max_workers=5)  # Adjust max_workers as needed

    async def run_in_executor(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))

    async def create_customer(self, username, source):
        customer = await self.run_in_executor(
            stripe.Customer.create,
            metadata={
                'online_username': username,
                'username_from': source
            }
        )
        return customer.id

    async def create_payment_link(self, user_id, price_id, customer_id=None):
        payment_link = await self.run_in_executor(
            stripe.PaymentLink.create,
            line_items=[{'price': price_id, 'quantity': 1}],
            metadata={'user_id': str(user_id), 'customer_id': customer_id or ''}
        )
        return payment_link.url

    @staticmethod
    def get_default_pricing():
        return config['credits']['stripe']['priceId']

    @staticmethod
    def get_credit_per_usd():
        return config['credits']['stripe']['credits_per_dollar']

    async def get_credit_amount_per_price_id(self, price_id):
        price_object = await self.run_in_executor(stripe.Price.retrieve, price_id)
        usd_amount = price_object.unit_amount / 100  # Stripe amounts are in cents
        return self.get_total_credit_amount(usd_amount)

    @staticmethod
    def get_total_credit_amount(usd_amount):
        return usd_amount * StripeIntegration.get_credit_per_usd()

    async def verify_payment_links_job(self):
        unconfirmed_payments = await Payment.query.filter_by(confirmed_at=None, type='stripe_payment_link').all()
        unconfirmed_links = [payment.txid for payment in unconfirmed_payments]

        events = await self.run_in_executor(stripe.Event.list)

        for event in events.auto_paging_iter():
            if event['type'] != "checkout.session.completed":
                continue

            payment_link = event['data']['object'].get('payment_link')
            if payment_link and payment_link in unconfirmed_links:
                link = await self.run_in_executor(stripe.PaymentLink.retrieve, payment_link)
                user_id = link.metadata.get('user_id')
                total_usd = event['data']['object'].get('amount_total') / 100
                credits = self.get_total_credit_amount(total_usd)

                user = await User.query.get(user_id)
                payment_to_confirm = next((p for p in unconfirmed_payments if p.txid == payment_link), None)

                if payment_to_confirm:
                    user.credits += credits
                    await user.save()

                    payment_to_confirm.confirmed_at = datetime.now()
                    await payment_to_confirm.save()

                    discord_id = user.discord_id

    async def verify_payment_links(self):
        await self.verify_payment_links_job()

    async def discord_balance_prompt(user_id, username):
        async with AsyncSessionLocal() as session:
            user = await session.execute(select(User).where(User.discordID == user_id))
            user = user.scalars().first()
            if user is None:
                user = User(username=username, discordID=user_id, credits=100)
                session.add(user)
                await session.commit()
            return user.credits
