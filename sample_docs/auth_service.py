"""
auth_service.py

Synthetic authentication & session management module.
Test fixture for AST-based code chunking + RAG evaluation.
"""

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 3600
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 900
PASSWORD_MIN_LENGTH = 10
TOKEN_BYTE_LENGTH = 32
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")


class Role(Enum):
    """User role levels, ordered by privilege."""
    GUEST = auto()
    MEMBER = auto()
    MODERATOR = auto()
    ADMIN = auto()


class AuthError(Exception):
    """Base class for authentication failures."""
    pass


class InvalidCredentialsError(AuthError):
    """Raised when a username/password pair doesn't match."""
    pass


class AccountLockedError(AuthError):
    """Raised when an account is locked out due to failed attempts."""

    def __init__(self, username: str, unlock_at: datetime):
        self.username = username
        self.unlock_at = unlock_at
        super().__init__(f"Account {username} locked until {unlock_at.isoformat()}")


class SessionExpiredError(AuthError):
    """Raised when a session token has expired or is unknown."""
    pass


def is_valid_email(email: str) -> bool:
    """Return True if the string looks like a valid email address."""
    return bool(EMAIL_REGEX.match(email))


def generate_token(byte_length: int = TOKEN_BYTE_LENGTH) -> str:
    """Generate a URL-safe random token string."""
    return secrets.token_urlsafe(byte_length)


def generate_salt() -> str:
    """Generate a random hex salt for password hashing."""
    return secrets.token_hex(16)


def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using PBKDF2-HMAC-SHA256."""
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return base64.b64encode(derived).decode("ascii")


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify a password against a stored salt+hash using constant-time compare."""
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


def check_password_strength(password: str) -> list[str]:
    """Return a list of validation issues with a candidate password (empty if OK)."""
    issues = []
    if len(password) < PASSWORD_MIN_LENGTH:
        issues.append(f"must be at least {PASSWORD_MIN_LENGTH} characters")
    if not any(c.isupper() for c in password):
        issues.append("must contain an uppercase letter")
    if not any(c.islower() for c in password):
        issues.append("must contain a lowercase letter")
    if not any(c.isdigit() for c in password):
        issues.append("must contain a digit")
    if password.lower() in ("password", "12345678", "qwerty123"):
        issues.append("is too common")
    return issues


def mask_email(email: str) -> str:
    """Mask the local part of an email for display, e.g. jo***@example.com."""
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked = local[0] + "*"
    else:
        masked = local[:2] + "*" * (len(local) - 2)
    return f"{masked}@{domain}"


def role_at_least(role: Role, minimum: Role) -> bool:
    """Return True if `role` has at least as much privilege as `minimum`."""
    return role.value >= minimum.value


def parse_bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract a bearer token from an Authorization header value."""
    if not header_value:
        return None
    parts = header_value.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


@contextmanager
def audit_context(action: str, actor: str):
    """Context manager that logs entry/exit of a sensitive auth action."""
    started = time.time()
    logger.info("AUDIT start action=%s actor=%s", action, actor)
    try:
        yield
    except Exception:
        logger.exception("AUDIT failed action=%s actor=%s", action, actor)
        raise
    else:
        elapsed = time.time() - started
        logger.info("AUDIT ok action=%s actor=%s elapsed=%.3fs", action, actor, elapsed)


def rate_limited_attempts(attempt_log: dict, key: str, window_seconds: int, max_attempts: int) -> bool:
    """Return True if `key` is under the rate limit within the sliding window."""
    now = time.time()
    timestamps = attempt_log.setdefault(key, [])
    cutoff = now - window_seconds

    def _prune(ts_list):
        return [t for t in ts_list if t >= cutoff]

    timestamps[:] = _prune(timestamps)
    if len(timestamps) >= max_attempts:
        return False
    timestamps.append(now)
    return True


def legacy_permission_check(user_roles, resource, action, overrides=None, deny_by_default=True):
    # NOTE: legacy, predates the Role enum, still used by old admin panel.
    allowed_actions = {
        "read": ["GUEST", "MEMBER", "MODERATOR", "ADMIN"],
        "write": ["MEMBER", "MODERATOR", "ADMIN"],
        "delete": ["MODERATOR", "ADMIN"],
        "manage_users": ["ADMIN"],
    }
    if overrides and resource in overrides:
        rule = overrides[resource]
        if action in rule:
            return any(r in rule[action] for r in user_roles)
    if action not in allowed_actions:
        return not deny_by_default
    for role in user_roles:
        if role in allowed_actions[action]:
            return True
    return not deny_by_default


@dataclass
class User:
    """A registered user account."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: Role = Role.MEMBER
    created_at: datetime = field(default_factory=datetime.utcnow)
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

    def is_locked(self) -> bool:
        """Return True if the account is currently locked out."""
        return self.locked_until is not None and datetime.utcnow() < self.locked_until

    def register_failed_attempt(self) -> None:
        """Increment failed attempt count, locking the account if over threshold."""
        self.failed_attempts += 1
        if self.failed_attempts >= MAX_LOGIN_ATTEMPTS:
            self.locked_until = datetime.utcnow() + timedelta(seconds=LOCKOUT_DURATION_SECONDS)

    def reset_failed_attempts(self) -> None:
        """Clear failed attempt count and any lockout."""
        self.failed_attempts = 0
        self.locked_until = None


@dataclass
class Session:
    """An active login session for a user."""
    token: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = SESSION_TTL_SECONDS

    def is_expired(self) -> bool:
        """Return True if the session has passed its TTL."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def remaining_seconds(self) -> float:
        """Return the number of seconds left before expiry (0 if expired)."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return max(0.0, self.ttl_seconds - age)


class UserStore(ABC):
    """Abstract persistence interface for user records."""

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[User]:
        ...

    @abstractmethod
    def save(self, user: User) -> None:
        ...


class InMemoryUserStore(UserStore):
    """A dict-backed UserStore implementation, keyed by username."""

    def __init__(self):
        self._users: dict[str, User] = {}

    def get_by_username(self, username: str) -> Optional[User]:
        """Return the User with this username, or None."""
        return self._users.get(username)

    def save(self, user: User) -> None:
        """Insert or update a user record."""
        self._users[user.username] = user

    def all_usernames(self) -> list[str]:
        """Return a list of all registered usernames."""
        return list(self._users.keys())

    def delete(self, username: str) -> bool:
        """Delete a user by username, returning True if one was removed."""
        return self._users.pop(username, None) is not None


class SessionStore:
    """In-memory session token store with expiry sweeping."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, user_id: str, ttl_seconds: int = SESSION_TTL_SECONDS) -> Session:
        """Create and store a new session for a user."""
        session = Session(token=generate_token(), user_id=user_id, ttl_seconds=ttl_seconds)
        self._sessions[session.token] = session
        return session

    def get(self, token: str) -> Session:
        """Return the session for a token, raising if missing or expired."""
        session = self._sessions.get(token)
        if session is None:
            raise SessionExpiredError("Unknown session token")
        if session.is_expired():
            del self._sessions[token]
            raise SessionExpiredError("Session token has expired")
        return session

    def revoke(self, token: str) -> None:
        """Revoke (delete) a session token if present."""
        self._sessions.pop(token, None)

    def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all sessions belonging to a user, returning the count revoked."""
        to_revoke = [t for t, s in self._sessions.items() if s.user_id == user_id]
        for token in to_revoke:
            del self._sessions[token]
        return len(to_revoke)

    def sweep_expired(self) -> int:
        """Remove all expired sessions, returning the count removed."""
        expired = [t for t, s in self._sessions.items() if s.is_expired()]
        for token in expired:
            del self._sessions[token]
        return len(expired)


class AuthService:
    """High-level authentication service: registration, login, sessions."""

    def __init__(self, user_store: Optional[UserStore] = None, session_store: Optional[SessionStore] = None):
        self.user_store = user_store or InMemoryUserStore()
        self.session_store = session_store or SessionStore()

    @staticmethod
    def _build_user(username: str, email: str, password: str, role: Role) -> User:
        salt = generate_salt()
        return User(
            user_id=generate_token(8),
            username=username,
            email=email,
            password_hash=hash_password(password, salt),
            salt=salt,
            role=role,
        )

    def register(self, username: str, email: str, password: str, role: Role = Role.MEMBER) -> User:
        """Register a new user, validating email format and password strength."""
        if not is_valid_email(email):
            raise ValueError(f"Invalid email: {email}")
        issues = check_password_strength(password)
        if issues:
            raise ValueError(f"Weak password: {'; '.join(issues)}")
        if self.user_store.get_by_username(username) is not None:
            raise ValueError(f"Username already taken: {username}")
        user = self._build_user(username, email, password, role)
        self.user_store.save(user)
        logger.info("Registered new user %s with role %s", username, role.name)
        return user

    def login(self, username: str, password: str) -> Session:
        """Authenticate a user and return a new session, or raise AuthError."""
        user = self.user_store.get_by_username(username)
        if user is None:
            raise InvalidCredentialsError(f"No such user: {username}")
        if user.is_locked():
            raise AccountLockedError(username, user.locked_until)
        if not verify_password(password, user.salt, user.password_hash):
            user.register_failed_attempt()
            self.user_store.save(user)
            raise InvalidCredentialsError("Incorrect password")
        user.reset_failed_attempts()
        self.user_store.save(user)
        return self.session_store.create(user.user_id)

    def logout(self, token: str) -> None:
        """Revoke a session token."""
        self.session_store.revoke(token)

    def current_user(self, token: str) -> User:
        """Resolve the User for a session token, raising if invalid/expired."""
        session = self.session_store.get(token)
        for user in self._all_users():
            if user.user_id == session.user_id:
                return user
        raise SessionExpiredError("Session refers to unknown user")

    def _all_users(self):
        if isinstance(self.user_store, InMemoryUserStore):
            return self.user_store._users.values()
        return []

    def require_role(self, token: str, minimum: Role) -> User:
        """Return the current user if they meet a minimum role, else raise."""
        user = self.current_user(token)
        if not role_at_least(user.role, minimum):
            raise AuthError(f"User {user.username} lacks required role {minimum.name}")
        return user