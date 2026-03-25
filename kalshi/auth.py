import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


class KalshiAuth:
    """RSA-PSS signing for Kalshi API authentication."""

    def __init__(self, key_id: str, private_key_path: str) -> None:
        self.key_id = key_id
        pem_bytes = Path(private_key_path).read_bytes()
        key = serialization.load_pem_private_key(pem_bytes, password=None)
        if not isinstance(key, RSAPrivateKey):
            raise TypeError("Private key must be RSA")
        self.private_key: RSAPrivateKey = key

    def sign_request(self, method: str, path: str) -> dict[str, str]:
        """Sign a request and return Kalshi auth headers.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: Full API path, e.g. /trade-api/v2/portfolio/balance.

        Returns:
            Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP,
            and KALSHI-ACCESS-SIGNATURE headers.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}"

        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        encoded_signature = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": encoded_signature,
        }
