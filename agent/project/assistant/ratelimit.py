"""按 user_id 的令牌桶限流

★ 为什么是"按用户"而不是"全局"：
    全局限流保护的是【服务器自己】，但一个用户就能把全局额度吃光，
    其他用户被无辜牵连。按用户限流保护的是【公平】—— 两者最终都要有。
"""

import os
import threading
import time


class TokenBucket:
    """令牌桶：容量 capacity，每秒补 refill_rate 个令牌。

    ★ 用 time.monotonic() 而不是 time.time()：
      monotonic 单调递增，不受系统时间被改（NTP 校时 / 手动改）影响。
      用 time.time() 的话，时钟往回跳会算出负的 elapsed，桶会被加满。
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity                 # 新用户先给满桶（允许一次突发）
        self.updated_at = time.monotonic()

    def _refill(self, now: float) -> None:
        """按流逝的时间补令牌。★ 惰性计算：不靠定时器，读的时候才算。"""
        elapsed = now - self.updated_at
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.updated_at = now

    def try_consume(self, amount: float = 1.0):
        """尝试消耗 amount 个令牌。

        返回 (是否放行, 还需等待几秒)；放行时第二个值是 0.0。
        """
        now = time.monotonic()
        self._refill(now)

        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0

        # 差多少令牌 → 按补充速率换算成"还要等几秒"
        missing = amount - self.tokens
        return False, missing / self.refill_rate


class RateLimiter:
    """按 user_id 管理令牌桶"""

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets = {}

        # ★★ 这把锁是必须的，不是可选的：
        #    FastAPI 的【同步】def 端点跑在 Starlette 的线程池里（默认 40 个线程），
        #    所以 check() 会被【真并发】调用。
        #    没有锁的话会出现竞态：两个线程同时读到 tokens=1，各自判断"够"，
        #    各自扣减 → 实际放行了 2 个请求，限流被击穿。
        self._lock = threading.Lock()

    def check(self, user_id: str, amount: float = 1.0):
        with self._lock:
            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = TokenBucket(self.capacity, self.refill_rate)
                self._buckets[user_id] = bucket
            return bucket.try_consume(amount)

    def stats(self):
        """运维用：看看当前有多少个桶、各自还剩多少令牌。"""
        with self._lock:
            now = time.monotonic()
            out = {}
            for uid, b in self._buckets.items():
                b._refill(now)
                out[uid] = round(b.tokens, 2)
            return out


# 默认：桶容量 5（允许一次突发 5 个请求），每秒补 0.05 个（≈ 每 20 秒 1 个）
# ★ 值故意调得很小，方便验收时一眼看到 429。生产按业务调。
limiter = RateLimiter(
    capacity=float(os.getenv("RL_CAPACITY", "5")),
    refill_rate=float(os.getenv("RL_REFILL_PER_SEC", "0.05")),
)