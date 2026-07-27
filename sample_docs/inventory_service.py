"""
inventory_service.py

A synthetic e-commerce inventory & order management module.
Generated as a test fixture for AST-based code chunking + RAG evaluation.
Contains a deliberately diverse mix of: module-level constants, simple and
complex functions, nested functions, decorators, async functions, classes
with inheritance, properties, classmethods, staticmethods, dataclasses,
and a couple of "messy" edge cases (long functions, no docstrings, etc.)
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 300
MAX_RETRY_ATTEMPTS = 3
LOW_STOCK_THRESHOLD = 10
TAX_RATE = 0.13
SUPPORTED_CURRENCIES = ("USD", "CAD", "EUR", "GBP")


class OrderStatus(Enum):
    """Lifecycle states for an order."""
    PENDING = "pending"
    PAID = "paid"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class InventoryError(Exception):
    """Base exception for inventory-related failures."""
    pass


class InsufficientStockError(InventoryError):
    """Raised when an order requests more units than are available."""

    def __init__(self, sku: str, requested: int, available: int):
        self.sku = sku
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient stock for {sku}: requested {requested}, available {available}"
        )


def retry(max_attempts: int = MAX_RETRY_ATTEMPTS, backoff_seconds: float = 0.5):
    """Decorator that retries a function on exception with linear backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt, max_attempts, func.__name__, exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(backoff_seconds * attempt)
            raise last_exc
        return wrapper
    return decorator


def timed(func):
    """Decorator that logs execution time of the wrapped function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("%s took %.2fms", func.__name__, elapsed_ms)
        return result
    return wrapper


def compute_sha256(data: str) -> str:
    """Return the hex-encoded SHA256 digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def chunk_id_for(file_path: str, qualified_name: str, source: str) -> str:
    """Build a deterministic chunk id from file path, name, and source text."""
    raw = f"{file_path}::{qualified_name}::{source}"
    return compute_sha256(raw)[:16]


def normalize_sku(sku: str) -> str:
    """Uppercase and strip a SKU string, collapsing internal whitespace."""
    return "-".join(sku.strip().upper().split())


def calculate_tax(amount: float, rate: float = TAX_RATE) -> float:
    """Calculate tax owed on a given amount at the given rate."""
    return round(amount * rate, 2)


def calculate_total_with_tax(amount: float, rate: float = TAX_RATE) -> float:
    """Calculate the total amount including tax."""
    return round(amount + calculate_tax(amount, rate), 2)


def apply_discount(amount: float, percent_off: float) -> float:
    """Apply a percentage discount to an amount, floored at zero."""
    if not 0 <= percent_off <= 100:
        raise ValueError("percent_off must be between 0 and 100")
    discounted = amount * (1 - percent_off / 100)
    return max(0.0, round(discounted, 2))


def is_low_stock(quantity: int, threshold: int = LOW_STOCK_THRESHOLD) -> bool:
    """Return True if quantity is at or below the low-stock threshold."""
    return quantity <= threshold


def days_until_restock(current_date: datetime, restock_date: datetime) -> int:
    """Return the number of whole days between now and a restock date."""
    delta = restock_date - current_date
    return max(0, delta.days)


def batch(iterable, size: int):
    """Yield successive chunks of `size` from `iterable`."""
    batch_buf = []
    for item in iterable:
        batch_buf.append(item)
        if len(batch_buf) == size:
            yield batch_buf
            batch_buf = []
    if batch_buf:
        yield batch_buf


def merge_price_overrides(base_prices: dict, overrides: dict) -> dict:
    """Merge a dict of override prices onto a dict of base prices.

    Overrides win on conflict. Neither input dict is mutated.
    """
    merged = dict(base_prices)
    for sku, price in overrides.items():
        if price < 0:
            logger.warning("Ignoring negative override price for %s: %s", sku, price)
            continue
        merged[sku] = price
    return merged


def build_order_summary(order: "Order") -> dict:
    """Build a plain-dict summary of an order for API responses."""
    def _line_item_dict(item):
        return {
            "sku": item.sku,
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "subtotal": round(item.quantity * item.unit_price, 2),
        }

    subtotal = sum(item.quantity * item.unit_price for item in order.line_items)
    tax = calculate_tax(subtotal)
    return {
        "order_id": order.order_id,
        "status": order.status.value,
        "line_items": [_line_item_dict(i) for i in order.line_items],
        "subtotal": round(subtotal, 2),
        "tax": tax,
        "total": round(subtotal + tax, 2),
    }


def validate_currency(currency: str) -> str:
    """Validate that a currency code is supported, returning it uppercased."""
    upper = currency.upper()
    if upper not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {currency}")
    return upper


async def fetch_price_async(sku: str, client: Any) -> float:
    """Asynchronously fetch the current price for a SKU from a pricing client."""
    response = await client.get(f"/prices/{sku}")
    return float(response["price"])


async def fetch_prices_concurrently(skus: list[str], client: Any) -> dict[str, float]:
    """Fetch prices for multiple SKUs concurrently."""
    tasks = [fetch_price_async(sku, client) for sku in skus]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    prices = {}
    for sku, result in zip(skus, results):
        if isinstance(result, Exception):
            logger.error("Failed to fetch price for %s: %s", sku, result)
            continue
        prices[sku] = result
    return prices


def legacy_reconcile_inventory(warehouse_counts, system_counts, tolerance=0, dry_run=True, audit_log=None):
    # NOTE: legacy function, no docstring, kept for backwards compatibility.
    # Reconciles physical warehouse counts against system-of-record counts.
    discrepancies = {}
    for sku, physical_qty in warehouse_counts.items():
        system_qty = system_counts.get(sku, 0)
        diff = physical_qty - system_qty
        if abs(diff) > tolerance:
            discrepancies[sku] = {
                "physical": physical_qty,
                "system": system_qty,
                "diff": diff,
            }
            if audit_log is not None:
                audit_log.append(
                    f"{datetime.utcnow().isoformat()} DISCREPANCY {sku}: physical={physical_qty} system={system_qty} diff={diff}"
                )
            if not dry_run:
                system_counts[sku] = physical_qty
    for sku, physical_qty in warehouse_counts.items():
        if sku not in system_counts:
            continue
    missing_in_warehouse = [sku for sku in system_counts if sku not in warehouse_counts]
    for sku in missing_in_warehouse:
        discrepancies.setdefault(sku, {
            "physical": 0,
            "system": system_counts[sku],
            "diff": -system_counts[sku],
        })
    return discrepancies


@dataclass
class LineItem:
    """A single line item within an order."""
    sku: str
    quantity: int
    unit_price: float

    def subtotal(self) -> float:
        """Return quantity * unit_price for this line item."""
        return round(self.quantity * self.unit_price, 2)


@dataclass
class Order:
    """Represents a customer order composed of one or more line items."""
    order_id: str
    customer_id: str
    line_items: list[LineItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_line_item(self, item: LineItem) -> None:
        """Append a line item to the order."""
        self.line_items.append(item)

    def total_quantity(self) -> int:
        """Return the sum of quantities across all line items."""
        return sum(item.quantity for item in self.line_items)

    def mark_paid(self) -> None:
        """Transition the order to the PAID status."""
        if self.status != OrderStatus.PENDING:
            raise InventoryError(f"Cannot pay order in status {self.status}")
        self.status = OrderStatus.PAID

    def cancel(self, reason: str = "") -> None:
        """Cancel the order if it hasn't been fulfilled yet."""
        if self.status == OrderStatus.FULFILLED:
            raise InventoryError("Cannot cancel a fulfilled order")
        self.status = OrderStatus.CANCELLED
        logger.info("Order %s cancelled: %s", self.order_id, reason or "no reason given")


class InMemoryCache:
    """A tiny TTL-based in-memory cache, not thread-safe."""

    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for key, or None if missing/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store a value under key with an optional custom TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl
        self._store[key] = (value, time.time() + ttl)

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache if present."""
        self._store.pop(key, None)

    @property
    def size(self) -> int:
        """Return the number of entries currently cached."""
        return len(self._store)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()


class InventoryRepository:
    """Abstract base for inventory persistence backends."""

    def get_quantity(self, sku: str) -> int:
        raise NotImplementedError

    def set_quantity(self, sku: str, quantity: int) -> None:
        raise NotImplementedError

    def decrement(self, sku: str, amount: int) -> None:
        raise NotImplementedError


class InMemoryInventoryRepository(InventoryRepository):
    """A simple dict-backed InventoryRepository implementation."""

    def __init__(self, initial_stock: Optional[dict[str, int]] = None):
        self._stock = dict(initial_stock or {})

    def get_quantity(self, sku: str) -> int:
        """Return current quantity on hand for a SKU, defaulting to 0."""
        return self._stock.get(normalize_sku(sku), 0)

    def set_quantity(self, sku: str, quantity: int) -> None:
        """Set the absolute quantity on hand for a SKU."""
        if quantity < 0:
            raise ValueError("quantity cannot be negative")
        self._stock[normalize_sku(sku)] = quantity

    def decrement(self, sku: str, amount: int) -> None:
        """Decrement stock for a SKU, raising if insufficient."""
        normalized = normalize_sku(sku)
        current = self._stock.get(normalized, 0)
        if current < amount:
            raise InsufficientStockError(normalized, amount, current)
        self._stock[normalized] = current - amount

    def bulk_load(self, counts: dict[str, int]) -> None:
        """Load a full set of SKU->quantity counts, overwriting existing state."""
        self._stock = {normalize_sku(sku): qty for sku, qty in counts.items()}


class InventoryService:
    """High-level service coordinating orders, stock, and caching."""

    def __init__(self, repository: InventoryRepository, cache: Optional[InMemoryCache] = None):
        self.repository = repository
        self.cache = cache or InMemoryCache()
        self._orders: dict[str, Order] = {}

    @classmethod
    def with_in_memory_backend(cls, initial_stock: Optional[dict[str, int]] = None) -> "InventoryService":
        """Convenience constructor wiring up in-memory repo + cache."""
        repo = InMemoryInventoryRepository(initial_stock)
        return cls(repository=repo, cache=InMemoryCache())

    @staticmethod
    def generate_order_id(customer_id: str) -> str:
        """Generate a pseudo-unique order id from a customer id and timestamp."""
        stamp = int(time.time() * 1000)
        return f"ORD-{customer_id[:6].upper()}-{stamp}"

    @timed
    def check_availability(self, sku: str, quantity: int) -> bool:
        """Return True if at least `quantity` units of sku are available."""
        cache_key = f"avail:{sku}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            available = cached
        else:
            available = self.repository.get_quantity(sku)
            self.cache.set(cache_key, available)
        return available >= quantity

    @retry(max_attempts=3)
    def reserve_stock(self, sku: str, quantity: int) -> None:
        """Reserve (decrement) stock for a SKU, retrying transient failures."""
        self.repository.decrement(sku, quantity)
        self.cache.invalidate(f"avail:{sku}")

    def create_order(self, customer_id: str, items: list[tuple[str, int, float]]) -> Order:
        """Create a new order, reserving stock for each requested line item.

        `items` is a list of (sku, quantity, unit_price) tuples. If any item
        cannot be reserved, previously reserved items in this call are not
        rolled back automatically -- caller is responsible for compensation.
        """
        order_id = self.generate_order_id(customer_id)
        order = Order(order_id=order_id, customer_id=customer_id)
        for sku, quantity, unit_price in items:
            normalized = normalize_sku(sku)
            if not self.check_availability(normalized, quantity):
                raise InsufficientStockError(
                    normalized, quantity, self.repository.get_quantity(normalized)
                )
            self.reserve_stock(normalized, quantity)
            order.add_line_item(LineItem(sku=normalized, quantity=quantity, unit_price=unit_price))
        self._orders[order_id] = order
        logger.info("Created order %s with %d line items", order_id, len(order.line_items))
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Look up an order by id, or None if it doesn't exist."""
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str, restock: bool = True) -> None:
        """Cancel an order and optionally restock its line items."""
        order = self._orders.get(order_id)
        if order is None:
            raise InventoryError(f"No such order: {order_id}")
        order.cancel(reason="customer requested cancellation")
        if restock:
            for item in order.line_items:
                current = self.repository.get_quantity(item.sku)
                self.repository.set_quantity(item.sku, current + item.quantity)
                self.cache.invalidate(f"avail:{item.sku}")

    def low_stock_report(self, threshold: int = LOW_STOCK_THRESHOLD) -> list[dict]:
        """Return a report of all SKUs at or below the given stock threshold."""
        report = []
        if isinstance(self.repository, InMemoryInventoryRepository):
            for sku, qty in self.repository._stock.items():
                if is_low_stock(qty, threshold):
                    report.append({"sku": sku, "quantity": qty})
        return sorted(report, key=lambda r: r["quantity"])

    def export_orders_json(self) -> str:
        """Serialize all known orders to a JSON string of summaries."""
        summaries = [build_order_summary(o) for o in self._orders.values()]
        return json.dumps(summaries, indent=2)


class PricingEngine:
    """Computes final prices given base prices, discounts, and tax rules."""

    def __init__(self, base_prices: dict[str, float], discounts: Optional[dict[str, float]] = None):
        self.base_prices = base_prices
        self.discounts = discounts or {}

    def price_for(self, sku: str, currency: str = "USD") -> float:
        """Return the tax-inclusive, discount-applied price for a SKU."""
        validate_currency(currency)
        base = self.base_prices.get(normalize_sku(sku))
        if base is None:
            raise KeyError(f"No base price for SKU {sku}")
        discount_pct = self.discounts.get(normalize_sku(sku), 0)
        discounted = apply_discount(base, discount_pct)
        return calculate_total_with_tax(discounted)

    def bulk_price(self, skus: list[str], currency: str = "USD") -> dict[str, float]:
        """Return prices for a list of SKUs as a dict."""
        return {sku: self.price_for(sku, currency) for sku in skus}

    def apply_seasonal_discount(self, sku: str, percent_off: float, expires_in_days: int = 7) -> datetime:
        """Apply a temporary seasonal discount to a SKU and return its expiry."""
        self.discounts[normalize_sku(sku)] = percent_off
        return datetime.utcnow() + timedelta(days=expires_in_days)