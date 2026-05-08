from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

VALID_SPEAKERS = ("alaa", "hammad", "hanan", "khalil")


# Client -> Server messages


class InitMessage(BaseModel):
    """First message from client to select a speaker voice."""

    type: Literal["init"] = "init"
    speaker: str = Field(..., description="Speaker name: alaa, hammad, hanan, or khalil")

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_SPEAKERS:
            raise ValueError(f"Invalid speaker '{v}'. Must be one of: {', '.join(VALID_SPEAKERS)}")
        return v


class AudioChunkMessage(BaseModel):
    """Audio chunk sent from client to server."""

    type: Literal["audio_chunk"] = "audio_chunk"
    data: str = Field(..., description="Base64-encoded audio data (PCM or compressed)")
    format: str = Field(default="wav", description="Audio format hint (wav, mp3, opus, etc.)")


class AudioEndMessage(BaseModel):
    """Signal that client has finished sending audio chunks."""

    type: Literal["audio_end"] = "audio_end"


class TerminateMessage(BaseModel):
    """Request to close the WebSocket connection."""

    type: Literal["terminate"] = "terminate"


class InterruptMessage(BaseModel):
    """Interrupt the current generation pipeline."""

    type: Literal["interrupt"] = "interrupt"


# Server -> Client messages


class CompleteMessage(BaseModel):
    """Signal that the pipeline has completed processing."""

    type: Literal["complete"] = "complete"


class ErrorMessage(BaseModel):
    """Error message from server to client."""

    type: Literal["error"] = "error"
    error: str = Field(..., description="Error description")


# Union type for message parsing


ClientMessage = InitMessage | AudioChunkMessage | AudioEndMessage | TerminateMessage | InterruptMessage
ServerMessage = CompleteMessage | ErrorMessage


def parse_client_message(data: dict) -> ClientMessage:
    """Parse a client message from a dictionary.

    Args:
        data: Dictionary containing the message data.

    Returns:
        Parsed client message.

    Raises:
        ValueError: If the message type is unknown or invalid.
    """
    msg_type = data.get("type")

    if msg_type == "init":
        return InitMessage.model_validate(data)
    elif msg_type == "audio_chunk":
        return AudioChunkMessage.model_validate(data)
    elif msg_type == "audio_end":
        return AudioEndMessage.model_validate(data)
    elif msg_type == "terminate":
        return TerminateMessage.model_validate(data)
    elif msg_type == "interrupt":
        return InterruptMessage.model_validate(data)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
