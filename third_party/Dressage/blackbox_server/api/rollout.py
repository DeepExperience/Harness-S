from __future__ import annotations

from fastapi import APIRouter, Request

from blackbox_server.core.models import (
    PauseRequest,
    PauseResponse,
    RegisterRequest,
    RegisterResponse,
    ResumeRequest,
    ResumeResponse,
)


router = APIRouter()


@router.post("/v1/rollout/register", response_model=RegisterResponse)
async def register_rollout(request: Request, payload: RegisterRequest) -> RegisterResponse:
    server = request.app.state.server
    async with server.request_scope():
        response, _ = await server.register(payload)
    response.request_id = request.state.request_id
    return response


@router.post("/v1/rollout/pause", response_model=PauseResponse)
async def pause_rollout(request: Request, payload: PauseRequest) -> PauseResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.pause_generation(payload)
    response.request_id = request.state.request_id
    return response


@router.post("/v1/rollout/resume", response_model=ResumeResponse)
async def resume_rollout(request: Request, payload: ResumeRequest) -> ResumeResponse:
    server = request.app.state.server
    async with server.request_scope():
        response = await server.resume_generation(payload)
    response.request_id = request.state.request_id
    return response


@router.get("/v1/rollout/pause_state")
async def pause_state(request: Request) -> dict:
    server = request.app.state.server
    async with server.request_scope():
        return await server.pause_state()
