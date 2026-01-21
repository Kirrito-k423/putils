import os
import hashlib
import torch
from contextlib import contextmanager

# cache 案例
# cache = RolloutCache(
#     cache_dir="/tmp/rollout_cache",
#     enabled=True,
#     force_recompute=False,
# )

# inputs = (
#     caption_list,
#     sample_text_hidden_states,
#     sample_text_attention_mask,
#     sample_negative_text_hidden_states,
#     sample_negative_text_attention_mask,
# )

# with rollout_cache(cache, *inputs) as result:
#     if isinstance(result, dict):
#         imgs = result["imgs"]
#         all_latents = result["all_latents"]
#         all_log_probs = result["all_log_probs"]
#     else:
#         # 真正计算
#         imgs, all_latents, all_log_probs = self.sora_rollout.generate(*inputs)

#         cache.save(
#             result,
#             to_cpu_detached(
#                 {
#                     "imgs": imgs,
#                     "all_latents": all_latents,
#                     "all_log_probs": all_log_probs,
#                 }
#             ),
#         )

class RolloutCache:
    def __init__(
        self,
        cache_dir,
        enabled=True,
        force_recompute=False,
    ):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.force_recompute = force_recompute
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_inputs(self, *args):
        m = hashlib.sha1()

        for arg in args:
            if torch.is_tensor(arg):
                m.update(str(arg.shape).encode())
                m.update(str(arg.dtype).encode())
                m.update(str(arg.device.type).encode())

                # 关键修复点
                t = arg.detach()
                if t.dtype in (torch.bfloat16, torch.float16):
                    t = t.float()

                sample = t.flatten()[:16].cpu().numpy()
                m.update(sample.tobytes())

            elif isinstance(arg, (list, tuple)):
                m.update(str(len(arg)).encode())
                for x in arg[:3]:  # 防止 caption 很长
                    m.update(str(x).encode())
            else:
                m.update(str(arg).encode())

        return m.hexdigest()


    def _cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.pt")

    def load(self, key):
        path = self._cache_path(key)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        return None

    def save(self, key, value):
        path = self._cache_path(key)
        torch.save(value, path)

@contextmanager
def rollout_cache(cache: RolloutCache, *inputs):
    if not cache.enabled:
        yield None
        return

    key = cache._hash_inputs(*inputs)

    if not cache.force_recompute:
        cached = cache.load(key)
        if cached is not None:
            yield cached
            return

    # cache miss
    yield key

def to_cpu_detached(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    elif isinstance(obj, dict):
        return {k: to_cpu_detached(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_cpu_detached(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(to_cpu_detached(v) for v in obj)

    else:
        # str / int / float / None / PIL.Image 等
        return obj


