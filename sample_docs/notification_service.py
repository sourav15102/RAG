"""
notification_service.py

Synthetic notification/messaging module: templates, channels, delivery,
scheduling, and rate limiting. Test fixture for AST-based code chunking
+ RAG evaluation.
"""

import logging
import random
import re
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

MAX_SMS_LENGTH = 160
MAX_PUSH_TITLE_LENGTH = 60
DEFAULT_RETRY_DELAYS = (1, 5, 30, 120)
TEMPLATE_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


class Channel(Enum):
    """Supported delivery channels for a notification."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    """Priority level, affecting queueing and retry behavior."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class NotificationError(Exception):
    """Base exception for notification delivery failures."""
    pass


class TemplateRenderError(NotificationError):
    """Raised when a template cannot be rendered due to missing variables."""

    def __init__(self, template_name: str, missing_vars: list[str]):
        self.template_name = template_name
        self.missing_vars = missing_vars
        super().__init__(f"Template '{template_name}' missing vars: {missing_vars}")


class DeliveryFailedError(NotificationError):
    """Raised when a channel adapter fails to deliver after all retries."""
    pass


class UnsubscribedError(NotificationError):
    """Raised when attempting to notify a user who has opted out of a channel."""
    pass


def extract_template_vars(template_text: str) -> list[str]:
    """Return the list of {{var}} placeholder names found in a template."""
    return TEMPLATE_VAR_PATTERN.findall(template_text)


def render_template(template_text: str, context: dict) -> str:
    """Render a {{var}} style template against a context dict."""
    required = extract_template_vars(template_text)
    missing = [v for v in required if v not in context]
    if missing:
        raise TemplateRenderError("<inline>", missing)

    def _replace(match):
        return str(context[match.group(1)])

    return TEMPLATE_VAR_PATTERN.sub(_replace, template_text)


def truncate_for_sms(text: str, max_length: int = MAX_SMS_LENGTH) -> str:
    """Truncate text to fit SMS length limits, appending an ellipsis if cut."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "\u2026"


def truncate_push_title(title: str, max_length: int = MAX_PUSH_TITLE_LENGTH) -> str:
    """Truncate a push notification title to the platform's display limit."""
    if len(title) <= max_length:
        return title
    return title[: max_length - 3].rstrip() + "..."


def generate_idempotency_key(length: int = 24) -> str:
    """Generate a random idempotency key for deduplicating sends."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def is_quiet_hours(now: datetime, quiet_start_hour: int = 22, quiet_end_hour: int = 8) -> bool:
    """Return True if `now` falls within a quiet-hours window (wraps midnight)."""
    hour = now.hour
    if quiet_start_hour > quiet_end_hour:
        return hour >= quiet_start_hour or hour < quiet_end_hour
    return quiet_start_hour <= hour < quiet_end_hour


def next_retry_delay(attempt: int, delays: tuple = DEFAULT_RETRY_DELAYS) -> int:
    """Return the retry delay in seconds for a given (1-indexed) attempt number."""
    index = min(attempt - 1, len(delays) - 1)
    return delays[max(0, index)]


def priority_sort_key(notification: "Notification") -> tuple:
    """Sort key placing higher priority and earlier scheduled_at first."""
    return (-notification.priority.value, notification.scheduled_at)


def legacy_merge_recipient_lists(*lists, dedupe_by="email", case_insensitive=True):
    # NOTE: legacy helper from the old bulk-email tool, kept for a few callers.
    merged = []
    seen_keys = set()
    for recipient_list in lists:
        for recipient in recipient_list:
            key = recipient.get(dedupe_by, "")
            if case_insensitive and isinstance(key, str):
                key = key.lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(recipient)
    return merged


@dataclass
class Notification:
    """A single notification to be delivered through one channel."""
    notification_id: str
    channel: Channel
    recipient: str
    body: str
    title: Optional[str] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    delivered: bool = False

    def mark_delivered(self) -> None:
        """Mark this notification as successfully delivered."""
        self.delivered = True

    def register_attempt(self) -> None:
        """Increment the delivery attempt counter."""
        self.attempts += 1

    def is_ready(self, now: Optional[datetime] = None) -> bool:
        """Return True if the notification's scheduled time has passed."""
        now = now or datetime.utcnow()
        return now >= self.scheduled_at


@dataclass
class Template:
    """A reusable notification template for a specific channel."""
    name: str
    channel: Channel
    body_template: str
    title_template: Optional[str] = None

    def render(self, context: dict) -> "Notification":
        """Render this template with context into a draft Notification-like dict."""
        body = render_template(self.body_template, context)
        title = render_template(self.title_template, context) if self.title_template else None
        if self.channel == Channel.SMS:
            body = truncate_for_sms(body)
        if title and self.channel == Channel.PUSH:
            title = truncate_push_title(title)
        return {"channel": self.channel, "body": body, "title": title}


class ChannelAdapter(ABC):
    """Base class for a channel-specific delivery adapter."""

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        ...


class EmailAdapter(ChannelAdapter):
    """Simulated email delivery adapter."""

    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.sent_log: list[str] = []

    def send(self, notification: Notification) -> bool:
        """Attempt to send an email notification, simulating occasional failure."""
        if random.random() < self.failure_rate:
            logger.warning("Simulated email failure for %s", notification.recipient)
            return False
        self.sent_log.append(notification.notification_id)
        return True


class SmsAdapter(ChannelAdapter):
    """Simulated SMS delivery adapter, enforcing length limits."""

    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.sent_log: list[str] = []

    def send(self, notification: Notification) -> bool:
        """Attempt to send an SMS notification, truncating if needed."""
        notification.body = truncate_for_sms(notification.body)
        if random.random() < self.failure_rate:
            return False
        self.sent_log.append(notification.notification_id)
        return True


class WebhookAdapter(ChannelAdapter):
    """Simulated outbound webhook delivery adapter."""

    def __init__(self, endpoint_url: str, failure_rate: float = 0.0):
        self.endpoint_url = endpoint_url
        self.failure_rate = failure_rate
        self.sent_log: list[str] = []

    def send(self, notification: Notification) -> bool:
        """POST a notification payload to the configured webhook endpoint."""
        if random.random() < self.failure_rate:
            return False
        logger.debug("POST %s payload for %s", self.endpoint_url, notification.notification_id)
        self.sent_log.append(notification.notification_id)
        return True


class SubscriptionRegistry:
    """Tracks which recipients have opted out of which channels."""

    def __init__(self):
        self._opt_outs: dict[str, set[Channel]] = {}

    def opt_out(self, recipient: str, channel: Channel) -> None:
        """Record that a recipient has opted out of a channel."""
        self._opt_outs.setdefault(recipient, set()).add(channel)

    def opt_in(self, recipient: str, channel: Channel) -> None:
        """Remove a channel opt-out for a recipient, if present."""
        if recipient in self._opt_outs:
            self._opt_outs[recipient].discard(channel)

    def is_subscribed(self, recipient: str, channel: Channel) -> bool:
        """Return True if the recipient has NOT opted out of the channel."""
        return channel not in self._opt_outs.get(recipient, set())


class NotificationService:
    """Coordinates templates, adapters, subscriptions, and delivery retries."""

    def __init__(self):
        self.adapters: dict[Channel, ChannelAdapter] = {}
        self.templates: dict[str, Template] = {}
        self.subscriptions = SubscriptionRegistry()
        self._queue: list[Notification] = []
        self._dead_letter: list[Notification] = []

    def register_adapter(self, channel: Channel, adapter: ChannelAdapter) -> None:
        """Register a channel adapter for outbound delivery."""
        self.adapters[channel] = adapter

    def register_template(self, template: Template) -> None:
        """Register a reusable template by name."""
        self.templates[template.name] = template

    def enqueue_from_template(
        self,
        template_name: str,
        recipient: str,
        context: dict,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
    ) -> Notification:
        """Render a template and enqueue the resulting notification."""
        template = self.templates.get(template_name)
        if template is None:
            raise NotificationError(f"Unknown template: {template_name}")
        if not self.subscriptions.is_subscribed(recipient, template.channel):
            raise UnsubscribedError(f"{recipient} unsubscribed from {template.channel.value}")
        rendered = template.render(context)
        notification = Notification(
            notification_id=generate_idempotency_key(),
            channel=rendered["channel"],
            recipient=recipient,
            body=rendered["body"],
            title=rendered["title"],
            priority=priority,
            scheduled_at=scheduled_at or datetime.utcnow(),
        )
        self._queue.append(notification)
        return notification

    def _deliver_one(self, notification: Notification) -> bool:
        adapter = self.adapters.get(notification.channel)
        if adapter is None:
            raise DeliveryFailedError(f"No adapter registered for {notification.channel.value}")
        for attempt in range(1, len(DEFAULT_RETRY_DELAYS) + 2):
            notification.register_attempt()
            if adapter.send(notification):
                notification.mark_delivered()
                return True
            delay = next_retry_delay(attempt)
            logger.warning(
                "Delivery attempt %d failed for %s, retrying in %ds",
                attempt, notification.notification_id, delay,
            )
        return False

    def flush(self, now: Optional[datetime] = None) -> dict:
        """Attempt delivery of all ready, queued notifications in priority order."""
        now = now or datetime.utcnow()
        ready = [n for n in self._queue if n.is_ready(now)]
        ready.sort(key=priority_sort_key)
        delivered, failed = 0, 0
        for notification in ready:
            try:
                if self._deliver_one(notification):
                    delivered += 1
                else:
                    failed += 1
                    self._dead_letter.append(notification)
            except DeliveryFailedError as exc:
                logger.error(str(exc))
                failed += 1
                self._dead_letter.append(notification)
            self._queue.remove(notification)
        return {"delivered": delivered, "failed": failed, "remaining": len(self._queue)}

    def dead_letter_count(self) -> int:
        """Return the number of notifications that exhausted all retries."""
        return len(self._dead_letter)

    def requeue_dead_letters(self) -> int:
        """Move all dead-lettered notifications back onto the main queue."""
        count = len(self._dead_letter)
        for notification in self._dead_letter:
            notification.attempts = 0
            notification.delivered = False
            self._queue.append(notification)
        self._dead_letter.clear()
        return count