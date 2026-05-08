"""Marker LLM service that delegates to the Claude Agent SDK.

This routes Marker's `--use_llm` cleanup pass through `claude_agent_sdk.query`
(which uses the local `claude` CLI under the user's Max subscription) instead of
the Anthropic billing API.

Wire it up:
    marker_single foo.pdf --use_llm \\
        --llm_service scripts.marker_sdk_service.ClaudeAgentSdkService

Or programmatically: pass the dotted path as `llm_service` to PdfConverter and
set `config={"use_llm": True, ...}`.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Annotated, List

import anyio
import PIL.Image
from pydantic import BaseModel

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from marker.services import BaseService

logger = logging.getLogger(__name__)


class ClaudeAgentSdkService(BaseService):
    """Marker LLM service that calls Claude via the Agent SDK."""

    sdk_model: Annotated[
        str,
        "Model alias passed to ClaudeAgentOptions. Use 'sonnet'/'opus'/'haiku' or"
        " a full model name. None lets the SDK pick the default.",
    ] = None
    sdk_max_turns: Annotated[
        int, "Max agent turns per call. Cleanup tasks are single-turn."
    ] = 2

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block,  # marker.schema.blocks.Block
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self.timeout

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        images = self._normalize_images(image)

        for attempt in range(1, max_retries + 2):
            try:
                response_text = anyio.run(
                    self._query_async, prompt, images, schema_json, timeout
                )
                result = self.validate_response(response_text, response_schema)
                if result:
                    return result
                logger.warning(
                    "Schema validation failed (attempt %d/%d); retrying.",
                    attempt,
                    max_retries + 1,
                )
            except Exception as exc:
                logger.error("SDK call failed (attempt %d): %s", attempt, exc)

        return {}

    @staticmethod
    def _normalize_images(image) -> list[PIL.Image.Image]:
        if image is None:
            return []
        if isinstance(image, list):
            return image
        return [image]

    async def _query_async(
        self,
        prompt: str,
        images: list[PIL.Image.Image],
        schema_json: str,
        timeout: int | None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="marker_sdk_") as tmpdir:
            tmp = Path(tmpdir)
            image_paths = []
            for idx, img in enumerate(images):
                p = tmp / f"region_{idx}.png"
                img.save(p, format="PNG")
                image_paths.append(str(p))

            image_block = (
                "\n".join(f"  - {p}" for p in image_paths)
                if image_paths
                else "  (none)"
            )
            user_prompt = (
                f"{prompt}\n\n"
                f"Image regions to inspect (read each with the Read tool):\n"
                f"{image_block}\n\n"
                f"Respond with ONLY a JSON object that matches this schema "
                f"exactly. No prose, no code fences, no commentary.\n\n"
                f"Schema:\n{schema_json}"
            )

            options = ClaudeAgentOptions(
                allowed_tools=["Read"] if image_paths else [],
                permission_mode="bypassPermissions",
                system_prompt=(
                    "You are a JSON-only formatter for the Marker PDF cleanup pipeline. "
                    "When you respond, output ONLY a single JSON object that conforms to "
                    "the schema given by the user. No prose, no markdown fences."
                ),
                max_turns=self.sdk_max_turns,
                cwd=str(tmp),
                # Region images and their JSON encoding can exceed the SDK's
                # 1 MB default buffer on math-heavy pages.
                max_buffer_size=64 * 1024 * 1024,
                **({"model": self.sdk_model} if self.sdk_model else {}),
            )

            chunks: list[str] = []

            async def run() -> None:
                async for message in query(prompt=user_prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        for blk in message.content:
                            if isinstance(blk, TextBlock):
                                chunks.append(blk.text)

            if timeout:
                with anyio.fail_after(timeout):
                    await run()
            else:
                await run()

            return "".join(chunks).strip()

    def validate_response(self, response_text: str, schema: type[BaseModel]):
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: -3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            return schema.model_validate_json(text).model_dump()
        except Exception:
            try:
                return schema.model_validate_json(text.replace("\\", "\\\\")).model_dump()
            except Exception as exc:
                logger.warning("JSON validation failed: %s; payload head=%r", exc, text[:200])
                return None
