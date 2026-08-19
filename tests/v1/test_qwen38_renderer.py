from functools import lru_cache

import pytest
from renderers import RenderedTokens
from renderers.configs import Qwen36RendererConfig
from renderers.qwen36 import Qwen36Renderer

from verifiers.v1.clients.qwen38 import (
    QWEN38_MODEL_ID,
    Qwen38Renderer,
    _replace_system_block,
    create_qwen38_renderer_pool,
)

QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "values": {"type": "array"},
                },
                "required": ["key"],
            },
        },
    }
]


@lru_cache(maxsize=1)
def _tokenizer_snapshot() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=QWEN38_MODEL_ID,
        revision=QWEN38_REVISION,
        allow_patterns=[
            "chat_template*",
            "config.json",
            "*.jinja",
            "tokenizer*",
        ],
    )


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_tokenizer_snapshot())


def _official_ids(
    messages,
    *,
    tools=None,
    add_generation_prompt=False,
    enable_thinking=True,
    preserve_thinking=True,
    reasoning_effort="xhigh",
):
    return list(
        _tokenizer().apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            reasoning_effort=reasoning_effort,
        )
    )


@pytest.mark.parametrize(
    "messages,tools",
    [
        pytest.param([{"role": "user", "content": "Hello"}], None, id="no-system"),
        pytest.param(
            [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello"},
            ],
            None,
            id="system",
        ),
        pytest.param(
            [{"role": "user", "content": "Look up x."}],
            TOOLS,
            id="tools-no-system",
        ),
        pytest.param(
            [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Look up x."},
            ],
            TOOLS,
            id="tools-with-system",
        ),
    ],
)
def test_qwen38_xhigh_system_shapes_match_official_template(messages, tools):
    renderer = Qwen38Renderer(_tokenizer())

    actual = renderer.render_ids(
        messages,
        tools=tools,
        add_generation_prompt=True,
    )

    assert actual == _official_ids(
        messages,
        tools=tools,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize(
    "reasoning_effort,instruction",
    [
        (
            "xhigh",
            (
                "Reasoning effort is set to xhigh. Please think carefully through the "
                "task, validate key assumptions, consider plausible alternatives, and "
                "prioritize correctness, consistency, and clarity in the final answer."
            ),
        ),
        ("medium", ""),
        (
            "low",
            (
                "Reasoning effort is set to low. Keep your thinking brief and focused, "
                "moving directly to the conclusion without unnecessary elaboration."
            ),
        ),
    ],
)
def test_qwen38_reasoning_effort_matches_official_placement(
    reasoning_effort, instruction
):
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "Use a tool."},
    ]
    renderer = Qwen38Renderer(_tokenizer(), reasoning_effort=reasoning_effort)

    actual = renderer.render_ids(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )
    text = _tokenizer().decode(actual)

    assert actual == _official_ids(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if instruction:
        assert text.index(instruction) < text.index("# Tools")
    else:
        assert "Reasoning effort is set" not in text
    assert text.index("# Tools") < text.index("SYSTEM_SENTINEL")


def test_qwen38_tool_arguments_and_multiturn_reasoning_match_official_template():
    messages = [
        {"role": "user", "content": "Look up x."},
        {
            "role": "assistant",
            "reasoning_content": "I should call the tool.",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": {
                            "key": "x",
                            "enabled": True,
                            "values": [1, None],
                        },
                    }
                },
                {"function": {"name": "lookup", "arguments": ""}},
            ],
        },
        {"role": "tool", "content": "42"},
        {"role": "tool", "content": "empty arguments accepted"},
        {
            "role": "assistant",
            "reasoning_content": "The tool returned 42.",
            "content": "The answer is 42.",
        },
        {"role": "user", "content": "Check once more."},
    ]

    actual = Qwen38Renderer(_tokenizer()).render_ids(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )

    assert actual == _official_ids(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )


def test_qwen38_thinking_disabled_omits_xhigh_instruction_and_matches_official():
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Answer briefly."},
    ]
    renderer = Qwen38Renderer(
        _tokenizer(),
        Qwen36RendererConfig(enable_thinking=False, preserve_thinking=False),
    )

    actual = renderer.render_ids(messages, add_generation_prompt=True)

    assert actual == _official_ids(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=False,
    )
    assert "Reasoning effort is set" not in _tokenizer().decode(actual)


def test_qwen38_can_drop_historical_reasoning_like_the_official_template():
    messages = [
        {"role": "user", "content": "Question one"},
        {
            "role": "assistant",
            "reasoning_content": "Historical reasoning",
            "content": "Answer one",
        },
        {"role": "user", "content": "Question two"},
    ]
    config = Qwen36RendererConfig(enable_thinking=True, preserve_thinking=False)

    actual = Qwen38Renderer(_tokenizer(), config).render_ids(
        messages,
        add_generation_prompt=True,
    )

    assert actual == _official_ids(
        messages,
        add_generation_prompt=True,
        preserve_thinking=False,
    )


def test_qwen38_system_replacement_preserves_metadata_alignment():
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "reasoning_content": "Reasoning",
            "content": "Answer",
        },
    ]

    config = Qwen36RendererConfig(enable_thinking=True, preserve_thinking=True)
    baseline = Qwen36Renderer(_tokenizer(), config).render(
        messages,
        add_generation_prompt=True,
    )
    rendered = Qwen38Renderer(_tokenizer(), config).render(
        messages,
        add_generation_prompt=True,
    )

    lengths = {
        len(rendered.token_ids),
        len(rendered.message_indices),
        len(rendered.sampled_mask),
        len(rendered.is_content),
    }
    assert len(lengths) == 1

    system_positions = [
        index
        for index, message_index in enumerate(rendered.message_indices)
        if message_index == 0
    ]
    assert system_positions
    assert not any(rendered.sampled_mask[index] for index in system_positions)
    system_content = [
        rendered.token_ids[index]
        for index in system_positions
        if rendered.is_content[index]
    ]
    assert _tokenizer().decode(system_content) == "SYSTEM_SENTINEL"

    baseline_suffix = baseline.message_indices.index(1)
    rendered_suffix = rendered.message_indices.index(1)
    for field in ("token_ids", "message_indices", "sampled_mask", "is_content"):
        assert (
            getattr(rendered, field)[rendered_suffix:]
            == getattr(baseline, field)[baseline_suffix:]
        )


def test_qwen38_system_replacement_rejects_an_unexpected_qwen36_prefix():
    with pytest.raises(
        RuntimeError,
        match=r"Qwen3\.6 system-block rendering changed unexpectedly",
    ):
        _replace_system_block(
            RenderedTokens(token_ids=[1]),
            RenderedTokens(token_ids=[2]),
            RenderedTokens(token_ids=[3]),
        )


def test_qwen38_image_tokens_match_official_template(monkeypatch):
    messages = [
        {"role": "system", "content": "Inspect the image carefully."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "image": "unused"},
            ],
        },
    ]
    renderer = Qwen38Renderer(_tokenizer())

    def process_image(_part):
        output = {"pixel_values": "pixels", "image_grid_thw": "grid"}
        return None, output, 1, "image-hash"

    monkeypatch.setattr(renderer, "_process_image", process_image)

    rendered = renderer.render(messages, add_generation_prompt=True)
    assert rendered.token_ids == _official_ids(
        messages,
        add_generation_prompt=True,
    )
    assert rendered.multi_modal_data is not None
    placeholders = rendered.multi_modal_data.mm_placeholders["image"]
    assert len(placeholders) == 1
    placeholder = placeholders[0]
    assert placeholder.length == 1
    pad_id = _tokenizer().convert_tokens_to_ids("<|image_pad|>")
    assert [
        index for index, token_id in enumerate(rendered.token_ids) if token_id == pad_id
    ] == [placeholder.offset]
    pad_slice = slice(placeholder.offset, placeholder.offset + placeholder.length)
    assert rendered.token_ids[pad_slice] == [pad_id]
    assert rendered.message_indices[pad_slice] == [1]
    assert rendered.sampled_mask[pad_slice] == [False]
    assert rendered.is_content[pad_slice] == [True]
    assert rendered.multi_modal_data.mm_items["image"] == [
        {"pixel_values": "pixels", "image_grid_thw": "grid"}
    ]


def test_qwen38_rejects_image_in_system_message_like_official_template():
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Inspect carefully."},
                {"type": "image", "image": "unused"},
            ],
        },
        {"role": "user", "content": "Question"},
    ]

    with pytest.raises(ValueError, match="System message cannot contain images"):
        Qwen38Renderer(_tokenizer()).render(messages)


def test_qwen38_video_content_fails_with_prime_specific_error():
    messages = [{"role": "user", "content": [{"type": "video", "video": "unused"}]}]

    with pytest.raises(
        NotImplementedError,
        match=r"Prime Qwen3\.8 video content is intentionally unsupported",
    ):
        Qwen38Renderer(_tokenizer()).render(messages)


def test_qwen38_bridge_matches_full_multiturn_rerender():
    renderer = Qwen38Renderer(_tokenizer())
    prompt_messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Question one"},
    ]
    assistant = {
        "role": "assistant",
        "reasoning_content": "Reasoning one",
        "content": "Answer one",
    }
    next_user = {"role": "user", "content": "Question two"}
    prompt_ids = renderer.render_ids(prompt_messages, add_generation_prompt=True)
    full_turn = renderer.render_ids([*prompt_messages, assistant])
    completion_ids = full_turn[len(prompt_ids) :]
    stop_ids = set(renderer.get_stop_token_ids())
    last_stop = max(
        index for index, token_id in enumerate(completion_ids) if token_id in stop_ids
    )
    completion_ids = completion_ids[: last_stop + 1]

    bridged = renderer.bridge_to_next_turn(
        prompt_ids,
        completion_ids,
        [next_user],
    )

    assert bridged is not None
    assert bridged.token_ids == _official_ids(
        [*prompt_messages, assistant, next_user],
        add_generation_prompt=True,
    )


def test_qwen38_pool_defaults_to_prime_xhigh_and_preserved_thinking():
    pool = create_qwen38_renderer_pool(_tokenizer_snapshot(), None, size=1)

    assert pool.renderer_cls is Qwen38Renderer
    with pool.checkout() as renderer:
        assert renderer.reasoning_effort == "xhigh"
        assert renderer.config.enable_thinking is True
        assert renderer.config.preserve_thinking is True
        assert renderer.effective_thinking_retention == "all"


def test_qwen38_pool_applies_config_and_reasoning_effort():
    config = Qwen36RendererConfig(preserve_thinking=False, image_cache_max=7)
    pool = create_qwen38_renderer_pool(
        _tokenizer_snapshot(),
        config,
        size=1,
        chat_template_kwargs={
            "reasoning_effort": "low",
            "add_vision_id": True,
        },
    )

    with pool.checkout() as renderer:
        assert renderer.reasoning_effort == "low"
        assert renderer.config.enable_thinking is True
        assert renderer.config.preserve_thinking is False
        assert renderer.config.add_vision_id is True
        assert renderer.config.image_cache_max == 7
