"""Offline rebuild after configuring embeddings; stop service before running.

Atomically updates vectors for each tenant after all that tenant's API calls pass.
Original messages/images and state are untouched. Back up the DB before use.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_memory.config import Settings
from fusion_memory.provider import Provider
from fusion_memory.storage import Storage


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", help="Reindex one tenant; otherwise all tenants")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.embedding_enabled:
        raise SystemExit("Set EMBEDDING_ENABLED=true first")
    storage = Storage(settings.db_path)
    provider = Provider(settings)
    with sqlite3.connect(settings.db_path) as conn:
        users = [r[0] for r in conn.execute("SELECT DISTINCT user_id FROM records")]
    if args.user_id:
        users = [args.user_id]
    count = 0
    try:
        for user in users:
            records = storage.list_records(user)
            vectors = await provider.embed([r["raw_text"]+"\n"+json.dumps(r["structured"],ensure_ascii=False) for r in records])
            with sqlite3.connect(settings.db_path) as conn:
                conn.executemany("UPDATE records SET embedding=?, embedding_model=? WHERE user_id=? AND id=?",
                    [(json.dumps(v), settings.embedding_model, user, r["id"]) for r,v in zip(records,vectors,strict=True)])
            count += len(records)
        print(json.dumps({"reindexed": count, "model": settings.embedding_model}))
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
