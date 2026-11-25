"""Single-user loopback Chat Completions server for P100 (FP16, chunked prefill, prefix KV reuse).

Native Qwen tool template is parsed into standard tool_calls. No private model service is modified.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
import uuid


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--context", type=int, default=8192)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        quantization_config=quant,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    revision = json.loads((Path(args.model) / "source_manifest.json").read_text())["revision"]
    lock = threading.Lock()
    cache = {"ids": [], "past": None}
    identity = {
        "id": "Qwen3-4B-Instruct-2507",
        "revision": revision,
        "thinking": False,
        "context_window": args.context,
        "engine": "transformers-p100",
        "dtype": "nf4-double-quant/compute-float16",
        "prefix_cache": True,
        "prefill_chunk": 256,
    }

    def complete(body):
        if body.get("model") != identity["id"]:
            raise ValueError("Unknown model")
        if body.get("chat_template_kwargs", {}).get("enable_thinking", False):
            raise ValueError("This deployment is non-thinking")
        tools = body.get("tools")
        choice = body.get("tool_choice", "auto")
        prompt = tokenizer.apply_chat_template(
            body["messages"],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix = ""
        if choice == "required":
            prefix = '<tool_call>\n{"name": "'
        elif isinstance(choice, dict):
            prefix = (
                '<tool_call>\n{"name": '
                + json.dumps(choice["function"]["name"])
                + ', "arguments": '
            )
        ids = tokenizer.encode(prompt + prefix, add_special_tokens=False)
        limit = int(body.get("max_tokens", 512))
        if len(ids) + limit > args.context:
            raise ValueError("Context limit exceeded")
        common = 0
        for a, b in zip(cache["ids"], ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if cache["past"] is not None:
            cache["past"].crop(common)
        past = cache["past"] if common else None
        if not common:
            cache["past"] = None
        inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            for start in range(common, len(ids), 256):
                end = min(start + 256, len(ids))
                output = model(
                    input_ids=inputs[:, start:end],
                    attention_mask=torch.ones((1, end), device="cuda", dtype=torch.long),
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
                past = output.past_key_values
            generated = []
            fed = []
            eos = tokenizer.eos_token_id
            generator = torch.Generator(device="cuda").manual_seed(int(body.get("seed", 234)))
            temperature = float(body.get("temperature", 0))
            finish = "length"
            for _ in range(limit):
                logits = output.logits[0, -1].float()
                token = (
                    int(torch.argmax(logits))
                    if temperature <= 0
                    else int(
                        torch.multinomial(
                            torch.softmax(logits / temperature, dim=-1), 1, generator=generator
                        )
                    )
                )
                generated.append(token)
                if token == eos:
                    finish = "stop"
                    break
                output = model(
                    input_ids=torch.tensor([[token]], device="cuda"),
                    attention_mask=torch.ones(
                        (1, len(ids) + len(fed) + 1), device="cuda", dtype=torch.long
                    ),
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                )
                past = output.past_key_values
                fed.append(token)
        cache.update(ids=ids + fed, past=past)
        text = prefix + tokenizer.decode(generated, skip_special_tokens=False)
        if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
            text = text[: -len(tokenizer.eos_token)]
        calls = []
        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
            try:
                action = json.loads(match.group(1))
                if not isinstance(action["name"], str) or not isinstance(action["arguments"], dict):
                    continue
                calls.append(
                    {
                        "id": "call_" + uuid.uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": action["name"],
                            "arguments": json.dumps(action["arguments"], ensure_ascii=False),
                        },
                    }
                )
            except (ValueError, KeyError, TypeError):
                continue
        message = {"role": "assistant", "content": None if calls else text}
        if calls:
            message["tool_calls"] = calls
            finish = "tool_calls"
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": identity["id"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": len(generated),
                "total_tokens": len(ids) + len(generated),
                "prompt_tokens_details": {"cached_tokens": common},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "deployment": identity,
            "tool_constraint_prefix_tokens": len(
                tokenizer.encode(prefix, add_special_tokens=False)
            ),
        }

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(
                200,
                {"data": [identity], "object": "list"} if self.path == "/v1/models" else identity,
            )

        def do_POST(self):
            if self.path=='/admin/clear_cache':
                import gc
                with lock:
                    cache.update(ids=[],past=None);gc.collect();torch.cuda.empty_cache()
                self.send(200,{'cleared':True});return
            if self.path != "/v1/chat/completions":
                self.send(404, {"error": "Unknown endpoint"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    result = complete(body)
                self.send(200, result)
            except ValueError as e:
                self.send(400, {"error": str(e)})
            except torch.cuda.OutOfMemoryError:
                cache.update(ids=[], past=None)
                torch.cuda.empty_cache()
                self.send(
                    503, {"error": "GPU OOM; retry may succeed after resources are available"}
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                cache.update(ids=[], past=None)
                self.send(500, {"error": type(e).__name__ + ": " + str(e)})

    print("MODEL_SERVER_READY", json.dumps(identity), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
