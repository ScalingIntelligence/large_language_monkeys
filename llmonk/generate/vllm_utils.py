import time
import requests
import subprocess
import socket
from contextlib import contextmanager
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from llmonk.utils import GenerateScriptConfig


def gpus_to_cvd(gpus: list[int]):
    return "CUDA_VISIBLE_DEVICES=" + ",".join([str(x) for x in gpus])


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def wait_for_ping(
    port,
    popen: subprocess.Popen,
    retry_seconds=2,
    max_retries=500,
    ping_endpoint: str = "ping",
):
    # wait for the server to start, by /ping-ing it
    print(f"Waiting for server to start on port {port}...")
    for i in range(max_retries):
        try:
            requests.get(f"http://localhost:{port}/{ping_endpoint}")
            return
        except requests.exceptions.ConnectionError:
            if popen.poll() is not None:
                raise RuntimeError(
                    f"Server died with code {popen.returncode} before starting."
                )

            print(f"Server not yet started (attempt {i}) retrying...")
            time.sleep(retry_seconds)

    raise RuntimeError(f"Server not started after {max_retries} attempts.")


@contextmanager
def vllm_manager(config: "GenerateScriptConfig"):

    gpus = config.gpus
    if isinstance(gpus, int):
        gpus = [gpus]
    elif isinstance(gpus, str):
        gpus = [int(gpu) for gpu in gpus.split(",")]

    model = config.model
    if config.vllm_args is None:
        args = ""
    else:
        str_args = [str(arg) for arg in config.vllm_args]
        args = " ".join(str_args)

    port = find_free_port()

    vllm_command = f"""python llmonk/generate/vllm_server.py \
        --model {model} \
        --port {port} \
        {args}"""

    if gpus is not None:
        vllm_command = (
            f"{gpus_to_cvd(gpus)} {vllm_command} --tensor-parallel-size {len(gpus)}"
        )

    print(f"Starting vllm server with command: {vllm_command}")
    vllm_process = subprocess.Popen(vllm_command, shell=True)
    print(f"Started vllm server with pid {vllm_process.pid}")

    try:
        wait_for_ping(port, vllm_process, max_retries=500)
        yield port
    finally:
        print(f"Killing vllm server (pid {vllm_process.pid})...")
        # kill the server and all its children
        subprocess.run(f"pkill -P -9 {vllm_process.pid}", shell=True)
        os.kill(vllm_process.pid, 9)

        print("Done killing vllm server.")
