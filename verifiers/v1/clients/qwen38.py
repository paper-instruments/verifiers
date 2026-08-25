"""Prime training renderer for Qwen3.8.

Prime's current task configuration does not schedule image or video tasks. The
renderer still preserves Qwen3.6's image path so adding an image cannot silently
change tokenization. Video remains intentionally unsupported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, cast

from renderers import RendererConfig
from renderers.base import (
    Message,
    RenderedTokens,
    ToolSpec,
    attribute_text_segments,
)
from renderers.configs import (
    Qwen35RendererConfig,
    Qwen36RendererConfig,
)
from renderers.qwen35 import (
    _TOOLS_FOOTER,
    _TOOLS_HEADER,
    _TOOLS_INSTRUCTIONS,
    _is_image_part,
    _is_video_part,
)
from renderers.qwen36 import Qwen36Renderer

QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"
_QWEN38_CACHE_COMPONENT = "models--Qwen--Qwen3.8-27B"
ReasoningEffort = Literal["xhigh", "medium", "low"]
_REASONING_INSTRUCTIONS: dict[ReasoningEffort, str] = {
    "xhigh": (
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
    ),
    "medium": "",
    "low": (
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
        "directly to the conclusion without unnecessary elaboration."
    ),
}


def is_qwen38_model(model_name: str) -> bool:
    """Recognize the canonical model ID and its Hugging Face snapshot path."""
    return model_name == QWEN38_MODEL_ID or _QWEN38_CACHE_COMPONENT in model_name.split(
        "/"
    )


def _validate_reasoning_effort(value: object) -> ReasoningEffort:
    if not isinstance(value, str) or value not in _REASONING_INSTRUCTIONS:
        raise ValueError("Qwen3.8 reasoning_effort must be xhigh, medium, or low.")
    return cast(ReasoningEffort, value)


class Qwen38Renderer(Qwen36Renderer):
    """Qwen3.6 plus Qwen3.8 reasoning instructions and explicit media policy.

    Prime defaults to ``xhigh``. Images retain the inherited Qwen3.6 rendering
    path for parity, although Prime does not currently schedule image tasks.
    Video is intentionally unsupported.
    """

    def __init__(
        self,
        tokenizer,
        config: Qwen36RendererConfig | None = None,
        *,
        reasoning_effort: ReasoningEffort = "xhigh",
    ) -> None:
        resolved_config = config or Qwen36RendererConfig(
            enable_thinking=True,
            preserve_thinking=True,
        )
        super().__init__(
            tokenizer,
            cast(Qwen35RendererConfig, resolved_config),
        )
        self.reasoning_effort = _validate_reasoning_effort(reasoning_effort)

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        _validate_media(messages)

        rendered = super().render(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

        # Qwen3.6 does not expose a system-block rendering hook. Rewriting the
        # input system message would also put this instruction after the tools,
        # while copying its full render loop would duplicate the much larger
        # assistant, tool, image, mask, and attribution implementation. Render
        # through Qwen3.6 and replace only its independently different prefix;
        # _replace_system_block verifies that the expected prefix is present.
        qwen36_block = self._render_system_block(
            messages,
            tools,
            instruction="",
            omit_empty_system=False,
        )
        qwen38_block = self._render_system_block(
            messages,
            tools,
            instruction=(
                _REASONING_INSTRUCTIONS[self.reasoning_effort]
                if self.config.enable_thinking
                else ""
            ),
            omit_empty_system=True,
        )
        return _replace_system_block(rendered, qwen36_block, qwen38_block)

    def _render_system_block(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        *,
        instruction: str,
        omit_empty_system: bool,
    ) -> RenderedTokens:
        first_is_system = messages[0].get("role") == "system"
        system_content = (
            self._render_content(messages[0].get("content")).strip()
            if first_is_system
            else ""
        )
        if (
            not tools
            and not instruction
            and (not first_is_system or (omit_empty_system and not system_content))
        ):
            return RenderedTokens()

        message_index = 0 if first_is_system else -1
        segments: list[tuple[str, bool]] = [("system\n", False)]
        if instruction:
            segments.append((instruction, False))
            if tools or system_content:
                segments.append(("\n\n", False))
        if tools:
            segments.append((_TOOLS_HEADER, False))
            segments.extend(
                ("\n" + json.dumps(tool, ensure_ascii=False), False) for tool in tools
            )
            segments.extend(
                [
                    (_TOOLS_FOOTER, False),
                    (_TOOLS_INSTRUCTIONS, False),
                ]
            )
            if system_content:
                segments.append(("\n\n", False))
        if system_content:
            segments.append((system_content, True))

        body = attribute_text_segments(self._tokenizer, segments)
        newline = self._encode("\n")
        token_ids = [self._im_start, *(token_id for token_id, _ in body), self._im_end]
        token_ids.extend(newline)
        indices = [message_index] * len(token_ids)
        sampled = [False] * len(token_ids)
        is_content = [False, *(value for _, value in body), False]
        is_content.extend([False] * len(newline))
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=is_content,
        )


def _validate_media(messages: list[Message]) -> None:
    """Enforce the media restrictions in the official template and Prime path."""
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if _is_video_part(item):
                raise NotImplementedError(
                    "Prime Qwen3.8 video content is intentionally unsupported."
                )
            if index == 0 and message.get("role") == "system" and _is_image_part(item):
                raise ValueError("System message cannot contain images.")


def _replace_system_block(
    rendered: RenderedTokens,
    old: RenderedTokens,
    new: RenderedTokens,
) -> RenderedTokens:
    """Replace Qwen3.6's leading system block and keep token metadata aligned."""
    if old.token_ids == new.token_ids:
        return rendered
    if rendered.token_ids[: len(old.token_ids)] != old.token_ids:
        raise RuntimeError("Qwen3.6 system-block rendering changed unexpectedly.")

    offset_delta = len(new.token_ids) - len(old.token_ids)
    if rendered.multi_modal_data is not None:
        for placeholders in rendered.multi_modal_data.mm_placeholders.values():
            for placeholder in placeholders:
                placeholder.offset += offset_delta

    return replace(
        rendered,
        token_ids=new.token_ids + rendered.token_ids[len(old.token_ids) :],
        message_indices=(
            new.message_indices + rendered.message_indices[len(old.message_indices) :]
        ),
        sampled_mask=new.sampled_mask + rendered.sampled_mask[len(old.sampled_mask) :],
        is_content=new.is_content + rendered.is_content[len(old.is_content) :],
    )


def create_qwen38_renderer(
    tokenizer,
    config: RendererConfig | None,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Qwen38Renderer:
    """Build Prime's Qwen3.8 renderer through Verifiers' shared renderer pool."""
    config_values: dict[str, Any] = {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    if config is not None:
        config_values.update(
            config.model_dump(
                exclude={"name"},
                exclude_none=True,
                exclude_unset=True,
            )
        )

    template_values = dict(chat_template_kwargs or {})
    reasoning_effort = _validate_reasoning_effort(
        template_values.pop("reasoning_effort", "xhigh")
    )
    config_values.update(template_values)
    renderer_config = Qwen36RendererConfig(**config_values)
    return Qwen38Renderer(
        tokenizer,
        renderer_config,
        reasoning_effort=reasoning_effort,
    )
