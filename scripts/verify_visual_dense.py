"""Live two-candidate visual and dense retrieval check using synthetic fixtures.

Run on the service host with ENV_FILE pointing to its configuration. Adds only
an isolated visual-smoke tenant; never edits existing memories or vectors.
"""
import argparse
import asyncio
import base64
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import uuid

import httpx
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_memory.config import Settings
from fusion_memory.provider import Provider
from fusion_memory.retrieval import rank_memories
from fusion_memory.storage import Storage


def picture(color, shape, label):
    im = Image.new("RGB", (520, 260), "white")
    draw = ImageDraw.Draw(im)
    if shape == "rectangle":
        draw.rectangle((40, 50, 210, 220), fill=color)
    else:
        draw.ellipse((40, 50, 210, 220), fill=color)
    draw.text((260, 110), label, fill="black", font_size=36)
    buffer = BytesIO()
    im.save(buffer, format="PNG")
    raw = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    cfg = Settings.from_env()
    if not cfg.embedding_enabled or cfg.embedding_dimensions != 1024:
        raise SystemExit("This check requires active 1024-dimensional embeddings")
    store, provider = Storage(cfg.db_path), Provider(cfg)
    owner = "visual-smoke:" + uuid.uuid4().hex
    fixtures = [
        {"color": "red", "shape": "rectangle", "label": "BOX 42", "number": "42"},
        {"color": "blue", "shape": "circle", "label": "TAG 77", "number": "77"},
    ]
    headers = {"Authorization": "Bearer " + cfg.service_token} if cfg.service_token else {}
    result = {"user_id": owner, "tests": []}
    try:
        async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=240) as client:
            for i, item in enumerate(fixtures):
                item["uri"], item["sha256"] = picture(item["color"], item["shape"], item["label"])
                response = await client.post("/add", json={
                    "user_id": owner, "request_id": f"{owner}:{i}", "session_id": f"session:{i}",
                    "messages": [{"role": "user", "timestamp": 1704067200000 + i * 1000,
                                  "content": [{"type": "text", "text": "Reference photograph."},
                                              {"type": "image_url", "image_url": {"url": item["uri"]}}]}],
                })
                response.raise_for_status()
                assert response.json()["success"] is True
            records = store.list_records(owner)
            assert len(records) == 2
            for item in fixtures:
                record = next(r for r in records if item["sha256"] in r["image_hashes"])
                assert record["raw_text"] == "Reference photograph."
                # These identifiers exist only in pixels, not in the raw text.
                assert item["number"] in json.dumps(record["structured"], ensure_ascii=False)
                assert record["embedding_model"] == "text-embedding-v4"
                assert len(record["embedding"]) == 1024
                item["id"] = record["id"]
                response = await client.post("/search", json={
                    "user_id": owner, "top_k": 1,
                    "query": [{"type": "image_url", "image_url": {"url": item["uri"]}}],
                })
                response.raise_for_status()
                rows = response.json()["data"]
                assert len(rows) == 1 and rows[0]["id"] == item["id"]
                returned = [part["image_url"]["url"] for part in rows[0]["content"]
                            if part["type"] == "image_url"]
                assert item["uri"] in returned
                result["tests"].append({"case": item["label"], "pixel_only_ocr_extracted": True,
                                        "image_only_query_correct": True, "original_image_delivered": True})

        # A controlled in-memory ablation removes every lexical feature and
        # graph edge, leaving only real stored vectors and a real query vector.
        dense_records = copy.deepcopy(records)
        for record in dense_records:
            record["raw_text"], record["structured"] = "", {}
        queries = [f"A {x['color']} {x['shape']} with visible text {x['label']}." for x in fixtures]
        vectors = await provider.embed(queries)
        for item, vector in zip(fixtures, vectors, strict=True):
            ranked = rank_memories(dense_records, "", vector, 1, query_embedding_model="text-embedding-v4",
                                   neighbor_hops=0, graph_expansion=False, diversity=False)
            assert ranked and ranked[0][0]["id"] == item["id"]
        assert rank_memories(dense_records, "", None, 1, neighbor_hops=0, graph_expansion=False) == []
        result.update({"passed": True, "stored_dimensions": 1024,
                       "dense_only_correct": 2, "no_vector_control_empty": True})
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        store.close()
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
