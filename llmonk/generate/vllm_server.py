"""
Modified from https://github.com/vllm-project/vllm/blob/a132435204aac8506e41813f90d08ddf7eca43b2/vllm/entrypoints/api_server.py
"""

import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.inputs import TokensPrompt

from dataclasses import fields
from llmonk.utils import dataclass_to_dict

TIMEOUT_KEEP_ALIVE = 18000  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


def make_output(request_output: RequestOutput):
    out = {}
    # iterate over dataclass fields
    for field in fields(CompletionOutput):
        # get the field name
        field_name = field.name
        field_list = [getattr(o, field_name) for o in request_output.outputs]
        out[field_name] = field_list

    if out["logprobs"][0] is not None:
        condensed_logprobs = []
        for old_logprobs in out["logprobs"]:
            new_logprobs = []
            for logprob in old_logprobs:
                new_logprobs.append({k: v.logprob for k, v in logprob.items()})
            condensed_logprobs.append(new_logprobs)

        out["logprobs"] = condensed_logprobs

    out["request_id"] = request_output.request_id
    out["prompt"] = request_output.prompt
    out["prompt_token_ids"] = request_output.prompt_token_ids
    out["prompt_logprobs"] = request_output.prompt_logprobs

    return dataclass_to_dict(out)


@app.get("/ping")
async def ping() -> Response:
    """Ping the server."""
    return Response(status_code=200, content="pong")


@app.get("/max_batch_size")
async def max_batch_size() -> Response:
    """Get the maximum batch size."""
    return JSONResponse({"max_batch_size": engine_args.max_num_seqs})


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt", None)
    input_ids = request_dict.pop("input_ids", None)

    if prompt is None and input_ids is None:
        return Response(
            status_code=400, content="Prompt or input_ids must be provided."
        )

    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    if prompt is None:
        prompt = TokensPrompt(prompt_token_ids=input_ids)

    results_generator = engine.generate(
        inputs=prompt, sampling_params=sampling_params, request_id=request_id
    )

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            out = make_output(request_output)
            yield (json.dumps(out) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)

        final_output = request_output

    assert final_output is not None
    out = make_output(final_output)
    return JSONResponse(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
