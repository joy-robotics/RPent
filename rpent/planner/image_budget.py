"""Limit outbound API image history without mutating saved messages or artifacts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai import BinaryContent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ImageUrl, ModelMessage, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext

from rpent.utils.logging import get_logger

logger = get_logger("api.images")
OMITTED_IMAGE = (
    "[Earlier image omitted from this API request; original artifact is retained.]"
)


def is_image(value: Any) -> bool:
    """Identify SDK image inputs without treating audio or plain URLs as images."""
    return isinstance(value, ImageUrl) or (
        isinstance(value, BinaryContent) and value.media_type.startswith("image/")
    )


def image_count(messages: list[ModelMessage]) -> int:
    """Count image occurrences in the user/tool content sent by RPent."""
    total = 0
    for message in messages:
        for part in message.parts:
            if isinstance(part, (UserPromptPart, ToolReturnPart)):
                content = part.content
                items = content if isinstance(content, (list, tuple)) else [content]
                total += sum(is_image(item) for item in items)
    return total


def limit_images(
    messages: list[ModelMessage], maximum: int
) -> tuple[list[ModelMessage], int]:
    """Copy changed content, retaining newest images and all text/tool boundaries."""
    if type(maximum) is not int or maximum < 1:
        raise ValueError("API image budget must be a positive integer")
    excess = max(0, image_count(messages) - maximum)
    if not excess:
        return messages, 0
    remaining = excess
    output = []
    for message in messages:
        parts = []
        changed_message = False
        for part in message.parts:
            if not isinstance(part, (UserPromptPart, ToolReturnPart)) or not remaining:
                parts.append(part)
                continue
            content = part.content
            items = content if isinstance(content, (list, tuple)) else [content]
            new_items = []
            changed_part = False
            for item in items:
                if remaining and is_image(item):
                    new_items.append(OMITTED_IMAGE)
                    remaining -= 1
                    changed_part = True
                else:
                    new_items.append(item)
            if changed_part:
                replacement = (
                    tuple(new_items)
                    if isinstance(content, tuple)
                    else new_items
                    if isinstance(content, list)
                    else new_items[0]
                )
                parts.append(replace(part, content=replacement))
                changed_message = True
            else:
                parts.append(part)
        output.append(replace(message, parts=parts) if changed_message else message)
    return output, excess


class RequestImageBudget(AbstractCapability):
    """Bound images at the public request hook and expose auditable counters."""

    def __init__(self, maximum: int):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("API image budget must be a positive integer")
        self.maximum = maximum
        self.trimmed_requests = 0
        self.omitted_image_occurrences = 0
        self.maximum_images_sent = 0

    async def before_model_request(
        self, ctx: RunContext, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        messages, removed = limit_images(request_context.messages, self.maximum)
        self.maximum_images_sent = max(self.maximum_images_sent, image_count(messages))
        if removed:
            self.trimmed_requests += 1
            self.omitted_image_occurrences += removed
            logger.info(
                "Image history budget: omitted %d older images from this request; kept at most %d; original artifacts unchanged",
                removed,
                self.maximum,
            )
            return replace(request_context, messages=messages)
        return request_context

    def stats(self) -> dict[str, int]:
        """Return request-side changes, not a count of deleted artifact files."""
        return {
            "limit": self.maximum,
            "trimmed_requests": self.trimmed_requests,
            "omitted_image_occurrences": self.omitted_image_occurrences,
            "maximum_images_sent": self.maximum_images_sent,
        }
