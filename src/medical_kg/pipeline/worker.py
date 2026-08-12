from __future__ import annotations

import socket
import uuid


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

