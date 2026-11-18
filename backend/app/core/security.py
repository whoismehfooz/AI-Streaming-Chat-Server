from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import TypeAlias

from argon2.exceptions import InvalidHashError
from pwdlib import PasswordHash
from pwdlib.exceptions import UnknownHashError


SecretValue: TypeAlias = str | bytes


_MAX_PASSWORD_BYTES = 1024
_MIN_TOKEN_BYTES = 32
_MAX_TOKEN_BYTES = 128
_MAX_OPAQUE_TOKEN_LENGTH = 8192
_SHA256_HEX_LENGTH = hashlib.sha256().digest_size * 2


_password_hasher = PasswordHash.recommended()


def _validate_password(password: str) -> None:

    if not isinstance(password, str):
        raise TypeError("Password must be a string.")

    if not password:
        raise ValueError("Password cannot be empty.")

    if len(password.encode("utf-8")) > _MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password cannot exceed {_MAX_PASSWORD_BYTES} UTF-8 bytes."
        )


def hash_password(password: str) -> str:

    _validate_password(password)
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:

    _validate_password(password)

    if not isinstance(password_hash, str):
        raise TypeError("Password hash must be a string.")

    if not password_hash:
        return False

    try:
        return _password_hasher.verify(password, password_hash)
    except (UnknownHashError, InvalidHashError):
        return False


def verify_password_and_update(
    password: str,
    password_hash: str,
) -> tuple[bool, str | None]:

    _validate_password(password)

    if not isinstance(password_hash, str):
        raise TypeError("Password hash must be a string.")

    if not password_hash:
        return False, None

    try:
        return _password_hasher.verify_and_update(password, password_hash)
    except (UnknownHashError, InvalidHashError):
        return False, None


def generate_secure_token(*, nbytes: int = 32) -> str:

    if isinstance(nbytes, bool) or not isinstance(nbytes, int):
        raise TypeError("nbytes must be an integer.")

    if not _MIN_TOKEN_BYTES <= nbytes <= _MAX_TOKEN_BYTES:
        raise ValueError(
            f"nbytes must be between {_MIN_TOKEN_BYTES} and {_MAX_TOKEN_BYTES}."
        )

    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:

    if not isinstance(token, str):
        raise TypeError("Token must be a string.")

    if not token:
        raise ValueError("Token cannot be empty.")

    if len(token) > _MAX_OPAQUE_TOKEN_LENGTH:
        raise ValueError(
            f"Token cannot exceed {_MAX_OPAQUE_TOKEN_LENGTH} characters."
        )

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_compare(
    first: SecretValue,
    second: SecretValue,
) -> bool:

    if  isinstance(first, str) and isinstance(second, str):
        return hmac.compare_digest(first, second)

    if isinstance(first, bytes) and isinstance(second, bytes):
        return hmac.compare_digest(first, second)

    raise TypeError("Secret values must have the same type.")


def verify_token(token: str, token_hash: str) -> bool:

    if not isinstance(token, str):
        raise TypeError("Token must be a string.")

    if not isinstance(token_hash, str):
        raise TypeError("Token hash must be a string.")

    if not token or len(token) > _MAX_OPAQUE_TOKEN_LENGTH:
        return False

    if len(token_hash) != _SHA256_HEX_LENGTH:
        return False

    try:
        int(token_hash, 16)
    except ValueError:
        return False

    candidate_hash = hash_token(token)
    return constant_time_compare(candidate_hash, token_hash.lower())
