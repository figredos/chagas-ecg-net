from typing import Callable, Awaitable

import time
import json
import logging

from fastapi import Request
from fastapi.datastructures import State

from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request[State]], Awaitable[Response]],
    ) -> Response:
        request.app.state.metrics["request_count"] += 1

        start = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000

        request.app.state.metrics["total_latency_ms"] += duration
        if response.status_code >= 400:
            request.app.state.metrics["error_count"] += 1

        log_dict = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration,
        }

        logging.getLogger(__name__).info(json.dumps(log_dict))

        return response
