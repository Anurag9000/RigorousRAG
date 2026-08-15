"""Cryptographic manifest-attestation contracts with an optional HMAC reference signer.

Public-key/KMS signers can implement the same protocol. The reference HMAC signer is for
controlled deployments/tests where a shared secret is acceptable; secret bytes are never
serialized into attestations.
"""
from __future__ import annotations
import base64, hashlib, hmac, json, time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

def _text(v: Any, label: str, maximum: int=500) -> str:
    if not isinstance(v,str) or not v.strip() or len(v.strip())>maximum or "\x00" in v: raise ValueError(f"{label} is invalid")
    return v.strip()
def _sha(v: str) -> str:
    d=_text(v,"digest",64).lower()
    if len(d)!=64 or any(c not in "0123456789abcdef" for c in d): raise ValueError("digest must be SHA-256")
    return d
def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload,Mapping): raise ValueError("manifest must be a mapping")
    return json.dumps(dict(payload),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
class ManifestSigner(Protocol):
    @property
    def key_id(self)->str: ...
    @property
    def algorithm(self)->str: ...
    def sign(self,payload:bytes)->bytes: ...
class ManifestVerifier(Protocol):
    def verify(self,*,key_id:str,algorithm:str,payload:bytes,signature:bytes)->bool: ...
@dataclass(frozen=True)
class ManifestAttestation:
    subject_id:str; manifest_sha256:str; key_id:str; algorithm:str; signature_b64:str; signed_at:float
    def __post_init__(self):
        object.__setattr__(self,"subject_id",_text(self.subject_id,"subject_id",500)); object.__setattr__(self,"manifest_sha256",_sha(self.manifest_sha256)); object.__setattr__(self,"key_id",_text(self.key_id,"key_id",500)); object.__setattr__(self,"algorithm",_text(self.algorithm,"algorithm",100).lower())
        try: base64.b64decode(self.signature_b64,validate=True)
        except Exception as exc: raise ValueError("signature_b64 is invalid") from exc
        if float(self.signed_at)<0: raise ValueError("signed_at is invalid")
def attest_manifest(subject_id:str,manifest:Mapping[str,Any],signer:ManifestSigner)->ManifestAttestation:
    payload=canonical_manifest_bytes(manifest); digest=hashlib.sha256(payload).hexdigest(); signature=signer.sign(payload)
    if not isinstance(signature,bytes) or not signature: raise RuntimeError("signer returned an invalid signature")
    return ManifestAttestation(_text(subject_id,"subject_id",500),digest,_text(signer.key_id,"key_id",500),_text(signer.algorithm,"algorithm",100),base64.b64encode(signature).decode("ascii"),time.time())
def verify_attestation(attestation:ManifestAttestation,manifest:Mapping[str,Any],verifier:ManifestVerifier)->bool:
    payload=canonical_manifest_bytes(manifest)
    if hashlib.sha256(payload).hexdigest()!=attestation.manifest_sha256: return False
    return bool(verifier.verify(key_id=attestation.key_id,algorithm=attestation.algorithm,payload=payload,signature=base64.b64decode(attestation.signature_b64)))
class HMACSHA256Signer:
    algorithm="hmac-sha256"
    def __init__(self,*,key_id:str,secret:bytes):
        self._key_id=_text(key_id,"key_id",500)
        if not isinstance(secret,bytes) or len(secret)<32: raise ValueError("HMAC secret must contain at least 32 bytes")
        self._secret=bytes(secret)
    @property
    def key_id(self)->str:return self._key_id
    def sign(self,payload:bytes)->bytes:return hmac.new(self._secret,payload,hashlib.sha256).digest()
    def verify(self,*,key_id:str,algorithm:str,payload:bytes,signature:bytes)->bool:return key_id==self.key_id and algorithm==self.algorithm and hmac.compare_digest(self.sign(payload),signature)
__all__=["HMACSHA256Signer","ManifestAttestation","ManifestSigner","ManifestVerifier","attest_manifest","canonical_manifest_bytes","verify_attestation"]
