from collections import OrderedDict


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
