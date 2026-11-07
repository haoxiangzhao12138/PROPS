"""OpenAI-compatible async client for the judge (Gemma-4-31B-it) and the grounder (Qwen3.8-27B).

Serve both with vLLM (scripts/serve_judge.sh). Requests use the reference settings: temperature 0.2, thinking
off, images downscaled to `max_side` and sent as JPEG quality 88. Replies are cached on disk by request hash, so an
interrupted run resumes without new calls.
"""
import asyncio, base64, hashlib, io, json, random, re
from functools import lru_cache
from pathlib import Path

from PIL import Image


@lru_cache(maxsize=4096)
def image_url(path, max_side):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def content(parts, max_side):
    """str -> text part, Path -> image part, in order."""
    out = []
    for x in parts:
        if isinstance(x, Path):
            out.append({"type": "image_url", "image_url": {"url": image_url(str(x), max_side)}})
        elif x:
            out.append({"type": "text", "text": str(x)})
    return out


def parse_json(text):
    """The first JSON object or array in a reply (reasoning blocks and code fences removed)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for a, b in (("{", "}"), ("[", "]")):
        s, e = text.find(a), text.rfind(b)
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except ValueError:
                pass
    raise ValueError("no JSON in reply: " + text[:200])


class Client:
    def __init__(self, bases, model, concurrency=16, timeout=120, retries=8, max_tokens=2048, cache=None, key="local"):
        from openai import AsyncOpenAI
        bases = [b for b in (bases.split(",") if isinstance(bases, str) else bases) if b]
        assert bases, "no server URL"
        self.clis = [AsyncOpenAI(base_url=b, api_key=key, timeout=timeout, max_retries=0) for b in bases]
        self.model, self.retries, self.max_tokens, self._rr = model, retries, max_tokens, 0
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = Path(cache) if cache else None

    async def ask(self, system, parts, max_side, max_tokens=None, temperature=0.2):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content(parts, max_side)}]
        kw = {"temperature": temperature, "max_tokens": max_tokens or self.max_tokens}
        f = None
        if self.cache:
            h = hashlib.sha256(json.dumps([self.model, msgs, kw], sort_keys=True).encode()).hexdigest()
            f = self.cache / h[:2] / f"{h}.json"
            if f.exists():
                return parse_json(json.loads(f.read_text())["text"])
        last = None
        for a in range(self.retries):
            try:
                async with self.sem:
                    self._rr += 1
                    cli = self.clis[self._rr % len(self.clis)]
                    r = await cli.chat.completions.create(model=self.model, messages=msgs, **kw,
                                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                text = r.choices[0].message.content or ""
                out = parse_json(text)
                if f:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps({"text": text}))
                return out
            except Exception as e:  # noqa: BLE001  timeouts, 429, bad JSON
                last = e
                if "Error code: 400" in str(e):
                    break
                await asyncio.sleep(min(8, 0.5 * 2 ** a) + random.random())
        raise RuntimeError(f"request failed after {self.retries} tries: {last}")


def answer(r, key="answer"):
    """The answer field of a reply, stripped; a reply that is not an object counts as its own text."""
    return str(r.get(key, "") if isinstance(r, dict) else r).strip()
