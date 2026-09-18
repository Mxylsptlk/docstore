"""Layout-aware chunking.

Packs a page's blocks (in reading order) into size-bounded chunks, with one hard rule:
a statistic must never be separated from its label. Each stat's value block and its
nearest label block form an atomic unit that always lands in the same chunk. Every
chunk carries page + union-bbox provenance, a stable chunk_id, and an is_stat flag so
retrieval can warn on unverified numbers.
"""

from __future__ import annotations

from typing import Optional

from docstore.models import BBox, Block, Chunk, Stat, make_chunk_id


def _union_bbox(bboxes: list[BBox]) -> BBox:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def _looks_like_heading(text: str) -> bool:
    t = text.strip()
    return 0 < len(t) <= 60 and "\n" not in t and not t.endswith((".", ":", ";"))


def _build_atomic_groups(blocks: list[Block], stats: list[Stat]) -> list[list[int]]:
    """Group block indices so a stat value stays with its label. Returns a list of
    groups (each a list of block indices), covering every block exactly once, in order."""
    # Map each block index -> group id; start each block in its own group.
    group_of = list(range(len(blocks)))

    def _find_block(pred) -> Optional[int]:
        for i, b in enumerate(blocks):
            if pred(b.text):
                return i
        return None

    for stat in stats:
        vi = _find_block(lambda t, v=stat.value_text: v and v in t)
        li = _find_block(lambda t, l=stat.label: l and l in t)
        if vi is not None and li is not None and vi != li:
            # merge the two groups (use the smaller group id)
            keep, drop = sorted((group_of[vi], group_of[li]))
            group_of = [keep if g == drop else g for g in group_of]

    # Collapse to ordered groups.
    groups: list[list[int]] = []
    seen: dict[int, int] = {}
    for i in range(len(blocks)):
        g = group_of[i]
        if g not in seen:
            seen[g] = len(groups)
            groups.append([])
        groups[seen[g]].append(i)
    return groups


def chunk_page(
    *,
    doc_id: str,
    source_path: str,
    page: int,
    blocks: list[Block],
    stats: list[Stat],
    size: int,
    overlap: int,
) -> list[Chunk]:
    if not blocks:
        return []

    groups = _build_atomic_groups(blocks, stats)
    section = blocks[0].text.strip() if _looks_like_heading(blocks[0].text) else ""

    chunks: list[Chunk] = []
    cur_idxs: list[int] = []
    cur_len = 0
    char_cursor = 0

    def flush():
        nonlocal cur_idxs, cur_len, char_cursor
        if not cur_idxs:
            return
        texts = [blocks[i].text for i in cur_idxs]
        text = "\n".join(t for t in texts if t)
        bbox = _union_bbox([blocks[i].bbox for i in cur_idxs])
        char_start = char_cursor
        char_end = char_start + len(text)
        # which stats fall in this chunk?
        chunk_stats = [s for s in stats if s.value_text and s.value_text in text]
        is_stat = bool(chunk_stats)
        stat_verified = _fold_verified(chunk_stats)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc_id, page, char_start, char_end),
                doc_id=doc_id,
                source_path=source_path,
                page=page,
                bbox=bbox,
                section=section,
                char_start=char_start,
                char_end=char_end,
                text=text,
                is_stat=is_stat,
                stat_verified=stat_verified,
            )
        )
        char_cursor = char_end + 1
        cur_idxs = []
        cur_len = 0

    for group in groups:
        group_text_len = sum(len(blocks[i].text) for i in group)
        # If adding this group would overflow and we already have content, flush first.
        if cur_idxs and cur_len + group_text_len > size:
            flush()
        cur_idxs.extend(group)
        cur_len += group_text_len
        # A single group larger than size still forms its own chunk (never split a stat).
        if cur_len >= size:
            flush()

    flush()
    return chunks


def _fold_verified(chunk_stats: list[Stat]) -> Optional[bool]:
    if not chunk_stats:
        return None
    if any(s.verified is False for s in chunk_stats):
        return False
    if all(s.verified is True for s in chunk_stats):
        return True
    return None  # mixed / unverifiable
