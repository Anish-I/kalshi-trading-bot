"""Poll an order until it reaches a terminal state (executed, canceled, expired)."""

import logging
import time

from kalshi.types import OrderResponse

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"executed", "canceled", "expired"}


def poll_order_until_terminal(
    client,
    order_id: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 2.0,
) -> OrderResponse:
    """Poll a Kalshi order until it reaches a terminal state.

    Args:
        client: KalshiClient instance (needs .get_order method).
        order_id: The order ID to poll.
        timeout_s: Maximum time to wait before giving up.
        poll_interval_s: Time between polls.

    Returns:
        OrderResponse with final status. If timeout, returns last known state.
    """
    start = time.monotonic()
    last_response = OrderResponse(order_id=order_id, status="unknown")

    while time.monotonic() - start < timeout_s:
        try:
            data = client.get_order(order_id)
            last_response = OrderResponse.from_dict(data)

            if last_response.status in TERMINAL_STATES:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.info("Order %s reached %s in %dms",
                            order_id, last_response.status, elapsed_ms)
                return last_response

            logger.debug("Order %s status=%s, polling...", order_id, last_response.status)
        except Exception:
            logger.warning("Failed to poll order %s", order_id, exc_info=True)

        time.sleep(poll_interval_s)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.warning("Order %s timed out after %dms (last status=%s)",
                   order_id, elapsed_ms, last_response.status)
    return last_response
