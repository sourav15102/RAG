import pytest


@pytest.fixture
def sample_source():
    return """
import os

TIMEOUT = 30

class PaymentService:
    \"\"\"Handles payment processing.\"\"\"

    tax_rate: float = 0.13

    def __init__(self, gateway_url: str):
        \"\"\"Initialize with gateway URL.\"\"\"
        self.gateway_url = gateway_url
        self._session = None

    def process_payment(self, order: dict, user: dict) -> bool:
        \"\"\"Process a payment for an order.\"\"\"
        self._validate_order(order)
        charge = self._calculate_charge(order)
        if charge > user["balance"]:
            raise ValueError("Insufficient funds")
        return self._complete_payment(charge)

    def _validate_order(self, order: dict) -> None:
        if not order.get("items"):
            raise ValueError("Order has no items")

    def _calculate_charge(self, order: dict) -> float:
        subtotal = sum(item["price"] for item in order["items"])
        return subtotal * (1 + self.tax_rate)

    async def _complete_payment(self, charge: float) -> bool:
        response = await self._session.post(self.gateway_url, json={"amount": charge})
        return response.status == 200


def format_receipt(order: dict, charge: float) -> str:
    \"\"\"Generate a plain-text receipt.\"\"\"
    lines = [f"Order #{order['id']}"]
    for item in order["items"]:
        lines.append(f"  {item['name']}: ${item['price']:.2f}")
    lines.append(f"Total: ${charge:.2f}")
    return "\\n".join(lines)
"""
