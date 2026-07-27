from fastapi import Request

from src.monitoring.ab_testing import ABRouter


def get_model(request: Request) -> ABRouter:
    return request.app.state.ab_router
