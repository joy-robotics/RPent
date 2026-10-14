import asyncio
from copy import deepcopy

import pytest
from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import ImageUrl, ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel

from rpent.planner.image_budget import (
    OMITTED_IMAGE,
    RequestImageBudget,
    image_count,
    limit_images,
)


def test_image_limit_retains_newest_images_text_and_tool_identity():
    older = BinaryContent(data=b"old", media_type="image/png")
    latest = BinaryContent(data=b"new", media_type="image/jpeg")
    audio = BinaryContent(data=b"audio", media_type="audio/wav")
    messages = [
        ModelRequest(parts=[UserPromptPart(content=["task", older])]),
        ModelResponse(parts=[TextPart(content="choose tool")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="observe",
                    tool_call_id="call-1",
                    content=[
                        "state unchanged",
                        ImageUrl(url="https://example.invalid/image.png"),
                        audio,
                        latest,
                    ],
                )
            ]
        ),
    ]
    original = deepcopy(messages)
    limited, removed = limit_images(messages, 2)
    assert removed == 1 and image_count(limited) == 2
    assert image_count(messages) == 3 and messages == original
    assert limited[0].parts[0].content == ["task", OMITTED_IMAGE]
    assert limited[2].parts[0].tool_call_id == "call-1"
    assert limited[2].parts[0].content[-1] is latest
    assert audio in limited[2].parts[0].content
    assert "state unchanged" in limited[2].parts[0].content


def test_under_budget_returns_original_without_changes():
    messages = [ModelRequest(parts=[UserPromptPart(content="no images")])]
    limited, removed = limit_images(messages, 48)
    assert limited is messages and removed == 0


@pytest.mark.parametrize("maximum", [0, -1, True, 2.5])
def test_invalid_budgets_rejected(maximum):
    with pytest.raises(ValueError):
        limit_images([], maximum)
    with pytest.raises(ValueError):
        RequestImageBudget(maximum)


def test_scalar_tool_image_and_tuple_user_content_are_counted():
    image = BinaryContent(data=b"image", media_type="image/png")
    messages = [
        ModelRequest(
            parts=[ToolReturnPart(tool_name="view", content=image, tool_call_id="old")]
        ),
        ModelRequest(parts=[UserPromptPart(content=("latest", image))]),
    ]
    limited, removed = limit_images(messages, 1)
    assert removed == 1 and image_count(limited) == 1
    assert limited[0].parts[0].content == OMITTED_IMAGE
    assert messages[0].parts[0].content is image
    assert isinstance(limited[1].parts[0].content, tuple)


def test_request_hook_preserves_streaming_and_model_metadata():
    image = BinaryContent(data=b"image", media_type="image/png")
    messages = [ModelRequest(parts=[UserPromptPart(content=[image, image])])]
    context = ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings={"max_tokens": 256},
        model_request_parameters=ModelRequestParameters(),
        model_id="test-alias",
        streaming=True,
    )
    budget = RequestImageBudget(1)
    updated = asyncio.run(budget.before_model_request(None, context))
    assert updated.streaming and updated.model_id == "test-alias"
    assert (
        updated.model is context.model
        and updated.model_settings == context.model_settings
    )
    assert image_count(updated.messages) == 1 and image_count(context.messages) == 2
    assert budget.stats() == {
        "limit": 1,
        "trimmed_requests": 1,
        "omitted_image_occurrences": 1,
        "maximum_images_sent": 1,
    }
