"""Quiet the noisy per-request INFO logs from HTTP/SDK clients.

This keeps E2B's build-progress output (emitted via default_build_logger's
callback, not the logging module) while silencing the httpx / e2b.api /
huggingface request spam.
"""

import logging

_NOISY = (
    "httpx",
    "e2b.api",
    "e2b.api.client_sync",
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "datasets",
    "urllib3",
)


def quiet_logs(level: int = logging.WARNING) -> None:
    for name in _NOISY:
        logging.getLogger(name).setLevel(level)
