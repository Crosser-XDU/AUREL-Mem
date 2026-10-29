"""Run with ENV_FILE=...; reports capability only, never keys or response bodies."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_memory.config import Settings
from fusion_memory.provider import Provider, UpstreamError


async def main():
    settings = Settings.from_env()
    provider = Provider(settings)
    report = {"model": settings.model, "chat": "not_checked", "embedding": "disabled"}
    try:
        await provider.extract([{"role": "user", "content": "My bicycle is blue.", "timestamp": 1704067200000}])
        report["chat"] = "passed"
    except UpstreamError:
        report["chat"] = "failed: check quota, credentials and model access"
    if settings.embedding_enabled:
        try:
            vectors = await provider.embed(["memory provider check"])
            report["embedding"] = {"passed": True, "model": settings.embedding_model, "dimensions": len(vectors[0])}
        except UpstreamError:
            report["embedding"] = "failed: check embedding provider/model access"
    await provider.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["chat"] == "passed" and not str(report["embedding"]).startswith("failed") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
