"""OpenAI-compatible async client for the judge (Gemma-4-31B-it) and the grounder (Qwen3.8-27B).

Serve both with vLLM (scripts/serve_judge.sh). Requests use the reference settings: temperature 0.2, thinking
off, images downscaled to `max_side` and sent as JPEG quality 88. Replies are cached on disk by request hash and the
weights behind the server, so an interrupted run resumes without new calls.
"""
import asyncio, base64, hashlib, io, json, random, re
from collections import OrderedDict
from pathlib import Path

from PIL import Image


_URLS = OrderedDict()  # (sha1 of the file, max_side) -> data URL, the 1,024 most recent


def image_url(path, max_side, size=None):
    """JPEG data URL of an image, first resized to `size` if given (a drawn frame is scored at the frame size, as
    `props generate` and `props collect` save it). Cached by the sha1 of the file content, so a file that is written
    again (the Reuse crop is re-cut when grounding is retried) is encoded again."""
    from .data import sha1
    key = (sha1(path), max_side, tuple(size) if size else None)
    if key in _URLS:
        _URLS.move_to_end(key)
        return _URLS[key]
    im = Image.open(path).convert("RGB")
    if size and im.size != tuple(size):
        im = im.resize(tuple(size), Image.BICUBIC)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    _URLS[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    if len(_URLS) > 1024:
        _URLS.popitem(last=False)
    return _URLS[key]


def content(parts, max_side):
    """str -> text part, Path or (Path, frame size) -> image part, in order."""
    out = []
    for x in parts:
        if isinstance(x, tuple):
            out.append({"type": "image_url", "image_url": {"url": image_url(x[0], max_side, x[1])}})
        elif isinstance(x, Path):
            out.append({"type": "image_url", "image_url": {"url": image_url(x, max_side)}})
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


def _get(url, timeout=10):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001  a server that does not answer is recorded as unknown
        return None


def served(url):
    """{served model name: weights} of one OpenAI-compatible server, or None if it does not answer. vLLM reports the
    weights as `root` in /v1/models."""
    m = _get(url.rstrip("/") + "/models") if url else None
    if not m or not m.get("data"):
        return None
    return {d.get("id"): d.get("root") or d.get("id") for d in m["data"]}


def weights(models, name):
    """The weights served under `name`, with their revision when known: from the second served name <name>@<rev>
    that scripts/serve_judge.sh adds, or from the .props_rev file that scripts/setup.sh writes next to the weights."""
    if name not in models:
        return None
    root, rev = models[name], None
    for k in models:
        if k.startswith(name + "@") and models[k] == root:
            rev = k[len(name) + 1:]
    if rev is None and (Path(root) / ".props_rev").exists():
        rev = (Path(root) / ".props_rev").read_text().strip()
    return {"root": root, "revision": rev}


def endpoints(urls, model):
    """{url: {"model", "weights", "vllm", "answers", "serves"}} for a comma-separated list of servers."""
    out = {}
    for u in (urls.split(",") if isinstance(urls, str) else urls):
        if u:
            ms, v = served(u), _get(u.rstrip("/").removesuffix("/v1") + "/version")
            out[u] = {"model": model, "weights": weights(ms, model) if ms else None,
                      "vllm": v.get("version") if v else None, "answers": ms is not None,
                      "serves": sorted(ms) if ms else []}
    return out


def identity(urls, model):
    """(identity, known): what the replies of a list of servers depend on, the served model name, the weights, their
    revision and the vLLM version, and whether the revision and the version are both known. Every server must
    answer, serve `model`, and report the same weights and version as the others."""
    eps = endpoints(urls, model)
    for u, e in eps.items():
        if not e["answers"]:
            raise ConnectionError(f"no server answers at {u}; start it (scripts/serve_judge.sh) or fix the URL")
        if model not in e["serves"]:
            raise ValueError(f"{u} serves {', '.join(e['serves'])}, not {model}; pass the served name with "
                             "--judge-model / --grounder-model")
    ids = {(json.dumps(e["weights"], sort_keys=True), e["vllm"]) for e in eps.values()}
    if len(ids) > 1:
        raise ValueError(f"the servers of {model} do not report the same weights and version: "
                         + ", ".join(f"{u} {e['weights']} {e['vllm']}" for u, e in eps.items()))
    w, v = ids.pop()
    w = json.loads(w)
    return f"{model}@{w['root']}@{w['revision']}@vllm-{v}", bool(w["revision"] and v)


class Client:
    def __init__(self, bases, model, concurrency=16, timeout=120, retries=8, max_tokens=2048, cache=None, key="local",
                 ident=None):
        from openai import AsyncOpenAI
        bases = [b for b in (bases.split(",") if isinstance(bases, str) else bases) if b]
        assert bases, "no server URL"
        self.clis = [AsyncOpenAI(base_url=b, api_key=key, timeout=timeout, max_retries=0) for b in bases]
        self.model, self.retries, self.max_tokens, self._rr = model, retries, max_tokens, 0
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = Path(cache) if cache else None
        self.ident = ident or model  # model name and weights, so cached replies of other weights are not reused

    async def ask(self, system, parts, max_side, max_tokens=None, temperature=0.2):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content(parts, max_side)}]
        kw = {"temperature": temperature, "max_tokens": max_tokens or self.max_tokens}
        f = None
        if self.cache:
            h = hashlib.sha256(json.dumps([self.ident, msgs, kw], sort_keys=True).encode()).hexdigest()
            f = self.cache / h[:2] / f"{h}.json"
            from .data import read_json
            c = read_json(f)
            if c and "text" in c:
                return parse_json(c["text"])
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
                    from .data import write_json
                    f.parent.mkdir(parents=True, exist_ok=True)
                    write_json(f, {"text": text})
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
