import importlib.util
spec = importlib.util.spec_from_file_location("lru", r"H:\ai-learning\algorithm\01_lru_cache.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
c = m.LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)
assert c.get(2) == -1, "2 应被淘汰"
assert c.get(3) == 3
c.put(4, 4)
assert c.get(1) == -1, "1 应被淘汰"
assert c.get(3) == 3 and c.get(4) == 4
print("全部通过 —— 你的 LRU Cache 正确")