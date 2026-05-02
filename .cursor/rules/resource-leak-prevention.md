# Resource Leak Prevention

## Rule: Prevent memory leaks and resource leaks in long-lived Python services

Adapted from daimonos Rust memory safety checklist. Python's GC prevents dangling
references but not logical leaks — unbounded growth, leaked file handles, and
unclosed sessions still cause OOM and connection exhaustion.

## Pre-commit checklist for new code

### 1. Dicts/lists in long-lived objects

- [ ] Is there a max size with eviction (e.g. `collections.OrderedDict` with `popitem`)?
- [ ] Is the max configurable via env var or config, not hardcoded?
- [ ] Is there a test that inserts 2× the max and asserts `len() <= max`?

### 2. DB sessions (SQLAlchemy)

- [ ] Every `SessionLocal()` / `Session()` is inside `try: ... finally: db.close()`
      or an `async with` context manager?
- [ ] Error paths (early returns, exceptions) still close the session?

### 3. File handles and temp files

- [ ] Every `open()` uses a `with` statement?
- [ ] Temp files/dirs created during processing are cleaned up in a `finally` block?
- [ ] Tests assert temp files are deleted after the operation?

### 4. WebSocket / connection tracking

- [ ] Connection removal is in `finally:`, not just the happy-path disconnect handler?
- [ ] The connection list has an upper bound to prevent DoS?
- [ ] A double-disconnect (`.remove()` on missing item) is handled gracefully?

### 5. Async context managers (MCP clients, HTTP clients)

- [ ] Every `__aenter__()` has a matching `__aexit__()` in a `finally` or `async with`?
- [ ] The owning class implements `__aenter__`/`__aexit__` so callers can use `async with`?

## Patterns

### Bounded dict
```python
from collections import OrderedDict

MAX = int(os.getenv("CACHE_MAX_ENTRIES", "1024"))

class BoundedCache:
    def __init__(self, maxsize: int = MAX):
        self._data = OrderedDict()
        self._maxsize = maxsize

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)
```

### Safe disconnect
```python
# BAD — leaks on non-WebSocketDisconnect exceptions
except WebSocketDisconnect:
    manager.disconnect(ws)

# GOOD — always cleans up
finally:
    manager.disconnect(ws)
```

### Session lifecycle
```python
# BAD — early return skips close
db = SessionLocal()
if not valid:
    return {"error": "..."}  # session leaked
db.close()

# GOOD — always closes
db = SessionLocal()
try:
    if not valid:
        return {"error": "..."}
    ...
finally:
    db.close()
```
