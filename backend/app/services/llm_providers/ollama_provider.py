import os
import json
import logging
import httpx
from typing import Generator, Dict, Any
from app.core.config import settings
from app.core.exceptions import OllamaUnavailableException

logger = logging.getLogger("app.services.llm_provider.ollama")

class OllamaProvider:
    """
    Concrete implementation of LLMProvider targeting local Ollama models.
    """
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Generator[Dict[str, Any], None, None]:
        url = f"{settings.OLLAMA_API_URL}/api/chat"

        # OLD: no explicit model quantization specified, no max_tokens cap, no 
        # num_ctx tuning — relied on Ollama defaults, which are slower and less 
        # predictable in cost/latency. Replaced below with explicit tuned settings.
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        options = {
            "temperature": temperature,
            "num_predict": min(max_tokens, 300) if max_tokens else 300,
            "num_ctx": 4096
        }
        if gpu_available:
            options["num_gpu"] = 35  # Load layers onto GPU
        else:
            num_threads = os.cpu_count() or 4
            options["num_thread"] = num_threads

        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "options": options
        }

        logger.info(
            f"Initiating local streaming chat completion with Ollama API '{url}' "
            f"using model '{settings.OLLAMA_MODEL}'..."
        )

        try:
            # OLD: a runtime connection failure to Ollama during an actual chat request 
            # (as opposed to the startup check) was not specifically caught — it likely 
            # surfaced as an unhandled connection error, potentially crashing the 
            # request or returning an unclear 500. Replaced below with a specific 
            # catch and a clear, actionable error message.
            # with httpx.stream("POST", url, json=payload, timeout=300.0) as r: ...   [old version, kept for reference]
            try:
                with httpx.stream(
                    "POST",
                    url,
                    json=payload,
                    timeout=300.0  # 5 minutes timeout to allow for local model loading and prompt evaluation
                ) as r:
                    r.raise_for_status()
                    prompt_tokens = 0
                    completion_tokens = 0

                    for line in r.iter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield {"type": "token", "token": data["message"]["content"]}

                            if data.get("done", False):
                                prompt_tokens = data.get("prompt_eval_count", 0)
                                completion_tokens = data.get("eval_count", 0)
                        except Exception as json_err:
                            logger.warning(f"Error parsing Ollama line JSON: {json_err}")

                    # Yield usage statistics (Ollama local inference has zero USD cost)
                    yield {
                        "type": "usage",
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "cost": 0.0
                    }
            except (httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
                logger.error(f"Ollama runtime connection failure: {conn_err}")
                raise OllamaUnavailableException()
        except Exception as e:
            if not isinstance(e, OllamaUnavailableException):
                logger.error(f"Ollama Provider streaming failure: {e}")
            raise e
