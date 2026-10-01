"""Redis state client with in-memory fallback for local development and unit tests."""
import os
import time
from typing import Dict, Any, List, Optional, Set, Union

class InMemoryRedisMock:
    """
    Mock Redis client storing sorted sets, hashes, and sets in memory.
    Fully compatible with Redis-py API for offline testing and fallback modes.
    """

    def __init__(self):
        self.zsets: Dict[str, List[tuple]] = {}  # key -> list of (score, member)
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.sets: Dict[str, Set[str]] = {}

    def ping(self) -> bool:
        return True

    def _parse_score_bound(self, bound: Union[float, int, str]) -> tuple:
        """Parse Redis score boundary, including '(' for exclusive bounds and +/-inf."""
        if isinstance(bound, (float, int)):
            return float(bound), False
        bound_str = str(bound).strip().lower()
        if bound_str in ("-inf", "-infinity"):
            return -float("inf"), False
        if bound_str in ("+inf", "+infinity", "inf", "infinity"):
            return float("inf"), False
        if bound_str.startswith("("):
            return float(bound_str[1:]), True
        return float(bound_str), False

    # --- Sorted Sets (ZSET) ---

    def zadd(self, key: str, mapping: Dict[str, float]):
        if key not in self.zsets:
            self.zsets[key] = []
        
        # Remove existing members if updating score
        members_to_add = {str(m): float(s) for m, s in mapping.items()}
        self.zsets[key] = [(s, m) for s, m in self.zsets[key] if m not in members_to_add]

        for member, score in members_to_add.items():
            self.zsets[key].append((score, member))
        
        # Keep sorted by score
        self.zsets[key].sort(key=lambda x: x[0])
        return len(mapping)

    def zremrangebyscore(self, key: str, min_score: Union[float, str], max_score: Union[float, str]):
        if key not in self.zsets:
            return 0
        min_val, min_exclusive = self._parse_score_bound(min_score)
        max_val, max_exclusive = self._parse_score_bound(max_score)

        def in_range(score):
            lower_ok = (score > min_val) if min_exclusive else (score >= min_val)
            upper_ok = (score < max_val) if max_exclusive else (score <= max_val)
            return lower_ok and upper_ok

        original_len = len(self.zsets[key])
        self.zsets[key] = [x for x in self.zsets[key] if not in_range(x[0])]
        return original_len - len(self.zsets[key])

    def zcount(self, key: str, min_score: Union[float, str], max_score: Union[float, str]) -> int:
        if key not in self.zsets:
            return 0
        min_val, min_exclusive = self._parse_score_bound(min_score)
        max_val, max_exclusive = self._parse_score_bound(max_score)

        def in_range(score):
            lower_ok = (score > min_val) if min_exclusive else (score >= min_val)
            upper_ok = (score < max_val) if max_exclusive else (score <= max_val)
            return lower_ok and upper_ok

        return sum(1 for x in self.zsets[key] if in_range(x[0]))

    def zrangebyscore(self, key: str, min_score: Union[float, str], max_score: Union[float, str], withscores: bool = False) -> List[Any]:
        if key not in self.zsets:
            return []
        min_val, min_exclusive = self._parse_score_bound(min_score)
        max_val, max_exclusive = self._parse_score_bound(max_score)

        def in_range(score):
            lower_ok = (score > min_val) if min_exclusive else (score >= min_val)
            upper_ok = (score < max_val) if max_exclusive else (score <= max_val)
            return lower_ok and upper_ok

        matching = [x for x in self.zsets[key] if in_range(x[0])]
        if withscores:
            return [(x[1].encode("utf-8"), x[0]) for x in matching]
        return [x[1].encode("utf-8") for x in matching]

    def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, []))

    def zrem(self, key: str, *members) -> int:
        if key not in self.zsets:
            return 0
        to_remove = set(str(m) for m in members)
        orig = len(self.zsets[key])
        self.zsets[key] = [x for x in self.zsets[key] if x[1] not in to_remove]
        return orig - len(self.zsets[key])

    # --- Hashes ---

    def hset(self, key: str, mapping: Optional[Dict[str, Any]] = None, key_field: Optional[str] = None, value: Optional[Any] = None):
        if key not in self.hashes:
            self.hashes[key] = {}
        if mapping:
            for k, v in mapping.items():
                self.hashes[key][str(k)] = str(v)
            return len(mapping)
        elif key_field is not None and value is not None:
            self.hashes[key][str(key_field)] = str(value)
            return 1
        return 0

    def hget(self, key: str, field: str) -> Optional[bytes]:
        val = self.hashes.get(key, {}).get(str(field))
        return val.encode("utf-8") if val is not None else None

    def hgetall(self, key: str) -> Dict[bytes, bytes]:
        res = self.hashes.get(key, {})
        return {str(k).encode("utf-8"): str(v).encode("utf-8") for k, v in res.items()}

    def hdel(self, key: str, *fields) -> int:
        if key not in self.hashes:
            return 0
        count = 0
        for f in fields:
            if str(f) in self.hashes[key]:
                del self.hashes[key][str(f)]
                count += 1
        return count

    # --- Sets ---

    def sadd(self, key: str, *members) -> int:
        if key not in self.sets:
            self.sets[key] = set()
        added = 0
        for m in members:
            str_m = str(m)
            if str_m not in self.sets[key]:
                self.sets[key].add(str_m)
                added += 1
        return added

    def smembers(self, key: str) -> Set[bytes]:
        return {m.encode("utf-8") for m in self.sets.get(key, set())}

    def srem(self, key: str, *members) -> int:
        if key not in self.sets:
            return 0
        removed = 0
        for m in members:
            str_m = str(m)
            if str_m in self.sets[key]:
                self.sets[key].remove(str_m)
                removed += 1
        return removed

    def scard(self, key: str) -> int:
        return len(self.sets.get(key, set()))

    def sismember(self, key: str, member: str) -> bool:
        return str(member) in self.sets.get(key, set())

    # --- General Keys ---

    def delete(self, *keys) -> int:
        count = 0
        for k in keys:
            if k in self.zsets:
                del self.zsets[k]
                count += 1
            if k in self.hashes:
                del self.hashes[k]
                count += 1
            if k in self.sets:
                del self.sets[k]
                count += 1
        return count

    def exists(self, *keys) -> int:
        return sum(1 for k in keys if k in self.zsets or k in self.hashes or k in self.sets)

    def flushall(self):
        self.zsets.clear()
        self.hashes.clear()
        self.sets.clear()

    def flushdb(self):
        self.flushall()

    def close(self):
        pass

def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: Optional[int] = None,
    password: Optional[str] = None,
    force_mock: bool = False
):
    """
    Attempt connecting to live Redis; fall back cleanly to InMemoryRedisMock if unavailable.
    Priority: Arguments > Environment variables > Defaults.
    """
    if force_mock:
        return InMemoryRedisMock()

    target_host = host or os.environ.get("REDIS_HOST", "localhost")
    target_port = int(port or os.environ.get("REDIS_PORT", 6379))
    target_db = int(db or os.environ.get("REDIS_DB", 0))
    target_password = password or os.environ.get("REDIS_PASSWORD", None)

    try:
        import redis
        client = redis.Redis(
            host=target_host,
            port=target_port,
            db=target_db,
            password=target_password,
            socket_connect_timeout=0.1,
            socket_timeout=0.2
        )
        client.ping()
        return client
    except Exception:
        return InMemoryRedisMock()
