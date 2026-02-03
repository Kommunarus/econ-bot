from typing import Dict, Set
from ddgs import DDGS
import hashlib

import time


search_cache: Dict[str, str] = {}
set_uid: Set[str] = set()

def web_search(query: str, k: int = 10) -> str:
    uid = hashlib.md5(query.encode("utf-8")).hexdigest()

    if uid in set_uid:
        print('use cache')
        return search_cache[uid]

    ddgs = DDGS()

    results = []

    for i, r in enumerate(ddgs.text(query, max_results=k, timelimit="y")):
        title = r.get("title")
        snippet = r.get("body") or r.get("Text") or ""
        url = r.get("href")
        results.append(f"{i + 1}. {title}: {snippet} [source: {url}]")


    output = "\n".join(results)
    search_cache[uid] = output
    set_uid.add(uid)
    return output