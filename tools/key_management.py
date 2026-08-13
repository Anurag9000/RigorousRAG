"""Metadata and callback contract for externally managed encryption keys."""
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol


@dataclass(frozen=True)
class WrappedKey:
    provider: str
    key_id: str
    key_version: str
    region: str
    ciphertext: bytes


class KeyProvider(Protocol):
    def wrap(self, plaintext_key: bytes, context: Mapping[str, str]) -> WrappedKey: ...
    def unwrap(self, wrapped_key: WrappedKey, context: Mapping[str, str]) -> bytes: ...


class CallbackKeyProvider:
    def __init__(self, wrap: Callable[[bytes, Mapping[str, str]], WrappedKey],
                 unwrap: Callable[[WrappedKey, Mapping[str, str]], bytes]) -> None:
        if not callable(wrap) or not callable(unwrap):
            raise ValueError("key provider callbacks must be callable")
        self._wrap, self._unwrap = wrap, unwrap

    def wrap(self, plaintext_key: bytes, context: Mapping[str, str]) -> WrappedKey:
        value = self._wrap(plaintext_key, context)
        if not isinstance(value, WrappedKey):
            raise RuntimeError("key provider returned invalid wrapped-key metadata")
        return value

    def unwrap(self, wrapped_key: WrappedKey, context: Mapping[str, str]) -> bytes:
        value = self._unwrap(wrapped_key, context)
        if not isinstance(value, bytes) or not value:
            raise RuntimeError("key provider returned invalid key material")
        return value


__all__ = ["CallbackKeyProvider", "KeyProvider", "WrappedKey"]
