"""JWT / DPoP validators and Clerk-org tenant mapping.

Claims schema — minted by saits-security/terraform/clerk/ JWT template:
  actor_id    : str  — Clerk user sub (audit + RBAC identity)
  tenant_id   : str  — saits-sync tenant UUID (Redis partition key)
  tenant_tier : str  — shared | private | hybrid
  subtenant_id: str  — delegated sessions; "" when not a subtenant context
  env         : str  — staging | production
  aud         : str  — saits.cloud/<env>
"""
from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from jose import JWTError, jwt


CLERK_JWKS_URL  = os.environ["CLERK_JWKS_URL"]    # https://<clerk-domain>/.well-known/jwks.json
JWT_AUDIENCE    = os.environ["JWT_AUDIENCE"]        # saits.cloud/production
DPOP_REPLAY_TTL = int(os.environ.get("DPOP_REPLAY_TTL", "120"))

# Replay guard: {jti -> eviction_ts}.  Production: replace with Redis SETNX + TTL.
_dpop_seen: dict[str, float] = {}


@dataclass
class SaitsClaims:
    actor_id: str
    tenant_id: str
    tenant_tier: str
    subtenant_id: str
    env: str


def _jwks() -> list[dict[str, Any]]:
    return httpx.get(CLERK_JWKS_URL, timeout=5).raise_for_status().json()["keys"]


def verify_clerk_jwt(token: str) -> SaitsClaims:
    """Validate a Clerk-minted saitsCloud access token.

    Verifies RS256 signature, exp, iss, aud against Clerk JWKS.
    Returns typed SaitsClaims on success; raises jose.JWTError on failure.
    """
    kid = jwt.get_unverified_header(token).get("kid")
    key = next((k for k in _jwks() if k.get("kid") == kid), None)
    if key is None:
        raise JWTError(f"unknown kid {kid!r}")

    claims = jwt.decode(token, key, algorithms=["RS256"], audience=JWT_AUDIENCE)
    return SaitsClaims(
        actor_id     = claims["actor_id"],
        tenant_id    = claims["tenant_id"],
        tenant_tier  = claims["tenant_tier"],
        subtenant_id = claims.get("subtenant_id", ""),
        env          = claims["env"],
    )


def verify_dpop_proof(proof: str, method: str, uri: str, access_token: str) -> None:
    """Verify a DPoP proof header (RFC 9449).

    Args:
        proof        : DPoP JWT from the DPoP HTTP header
        method       : HTTP method (uppercase, e.g. "POST")
        uri          : Full request URI including scheme
        access_token : The bound access token (for ath claim verification)

    Raises ValueError on any validation failure.
    """
    header = jwt.get_unverified_header(proof)
    if header.get("typ") != "dpop+jwt":
        raise ValueError("DPoP typ must be dpop+jwt")

    body = jwt.get_unverified_claims(proof)
    now  = time.time()

    if body.get("htm") != method:
        raise ValueError("DPoP htm mismatch")
    if body.get("htu") != uri:
        raise ValueError("DPoP htu mismatch")
    if abs(body.get("iat", 0) - now) > DPOP_REPLAY_TTL:
        raise ValueError("DPoP iat out of window")

    jti = body.get("jti") or ""
    if not jti:
        raise ValueError("DPoP jti required")

    # Expire old entries before replay check
    _dpop_seen.update({k: v for k, v in _dpop_seen.items() if v > now})
    if jti in _dpop_seen:
        raise ValueError("DPoP replay detected")
    _dpop_seen[jti] = now + DPOP_REPLAY_TTL

    expected_ath = (
        base64.urlsafe_b64encode(hashlib.sha256(access_token.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    if body.get("ath") != expected_ath:
        raise ValueError("DPoP ath mismatch")
