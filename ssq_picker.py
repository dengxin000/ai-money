#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


RED_MIN = 1
RED_MAX = 33
BLUE_MIN = 1
BLUE_MAX = 16
WINDOW = 20


@dataclass(frozen=True)
class Draw:
    period: str
    red: tuple[int, ...]
    blue: int


@dataclass
class Stats:
    red_counts: dict[int, int]
    red_miss: dict[int, int]
    blue_counts: dict[int, int]
    blue_miss: dict[int, int]
    sum_mean: float
    sum_std: float
    last_sum: float
    last_high_count: int
    high_streak: int
    last_red: set[int]


def parse_draws(path: Path) -> list[Draw]:
    data = json.loads(path.read_text(encoding="utf-8"))
    draws: list[Draw] = []
    for item in data:
        red = tuple(sorted(int(x) for x in item["red"]))
        blue = int(item["blue"])
        period = str(item["period"])
        validate_draw(red, blue, period)
        draws.append(Draw(period=period, red=red, blue=blue))
    draws.sort(key=lambda d: d.period)
    return draws[-WINDOW:]


def validate_draw(red: Iterable[int], blue: int, period: str = "") -> None:
    red_list = list(red)
    if len(red_list) != 6:
        raise ValueError(f"{period}: red ball count must be 6")
    if len(set(red_list)) != 6:
        raise ValueError(f"{period}: red balls must be unique")
    if any(n < RED_MIN or n > RED_MAX for n in red_list):
        raise ValueError(f"{period}: red balls out of range")
    if blue < BLUE_MIN or blue > BLUE_MAX:
        raise ValueError(f"{period}: blue ball out of range")


def calc_stats(draws: list[Draw]) -> Stats:
    red_counts = {n: 0 for n in range(RED_MIN, RED_MAX + 1)}
    blue_counts = {n: 0 for n in range(BLUE_MIN, BLUE_MAX + 1)}

    for draw in draws:
        for n in draw.red:
            red_counts[n] += 1
        blue_counts[draw.blue] += 1

    red_miss = calc_miss(draws, is_red=True)
    blue_miss = calc_miss(draws, is_red=False)
    sums = [sum(draw.red) for draw in draws]
    sum_mean = statistics.mean(sums)
    sum_std = statistics.pstdev(sums) if len(sums) > 1 else 0.0
    return Stats(
        red_counts=red_counts,
        red_miss=red_miss,
        blue_counts=blue_counts,
        blue_miss=blue_miss,
        sum_mean=sum_mean,
        sum_std=sum_std,
        last_sum=sums[-1] if sums else 0.0,
        last_high_count=sum(1 for n in draws[-1].red if region(n) == 2) if draws else 0,
        high_streak=calc_high_streak(sums, sum_mean, sum_std),
        last_red=set(draws[-1].red) if draws else set(),
    )


def calc_high_streak(sums: list[int], mean: float, std: float) -> int:
    if not sums or std == 0:
        return 0
    threshold = mean + std * 0.6
    streak = 0
    for total in reversed(sums):
        if total >= threshold:
            streak += 1
        else:
            break
    return streak


def calc_miss(draws: list[Draw], *, is_red: bool) -> dict[int, int]:
    if is_red:
        pool = range(RED_MIN, RED_MAX + 1)
        getter = lambda d: d.red
    else:
        pool = range(BLUE_MIN, BLUE_MAX + 1)
        getter = lambda d: (d.blue,)

    miss = {n: len(draws) for n in pool}
    for offset, draw in enumerate(reversed(draws), start=0):
        seen = set(getter(draw))
        for n in seen:
            if miss[n] == len(draws):
                miss[n] = offset
    return miss


def region(n: int) -> int:
    if 1 <= n <= 11:
        return 0
    if 12 <= n <= 22:
        return 1
    return 2


def high_band(n: int) -> int:
    if 23 <= n <= 27:
        return 1
    if 29 <= n <= 33:
        return 2
    return 0


def red_pool(stats: Stats, size: int = 20) -> list[int]:
    items = list(range(RED_MIN, RED_MAX + 1))

    def score(n: int) -> float:
        count = stats.red_counts[n]
        miss = stats.red_miss[n]
        recent = 1.0 if n in stats.last_red else 0.0
        region_bonus = 0.6 if count > 0 else 0.0
        warm_miss = 1.4 if 2 <= miss <= 6 else 0.0
        middle_bias = 1.0 if region(n) == 1 else 0.0
        push_mode = stats.high_streak >= 2
        band = high_band(n)
        high_band_bias = 2.0 if band == 1 and not push_mode else 0.8 if band == 1 else 2.2 if band == 2 and push_mode else 0.15 if band == 2 else 0.0
        low_sum_bias = 0.4 if region(n) == 0 and not push_mode and miss <= 6 else 0.0
        sticky_penalty = (count - 3) * 0.5 if count >= 4 else 0.0
        return count * 2.8 + min(miss, 20) * 0.65 + warm_miss + middle_bias + high_band_bias + low_sum_bias + region_bonus - recent * 0.4 - sticky_penalty

    items.sort(key=lambda n: (score(n), -n), reverse=True)
    picked: list[int] = []
    region_seen = [0, 0, 0]

    for n in items:
        if len(picked) >= size:
            break
        r = region(n)
        if region_seen[r] >= 7:
            continue
        picked.append(n)
        region_seen[r] += 1

    if len(picked) < size:
        for n in items:
            if n not in picked:
                picked.append(n)
            if len(picked) >= size:
                break
    return sorted(picked)


def blue_pool(stats: Stats, size: int = 6) -> list[int]:
    items = list(range(BLUE_MIN, BLUE_MAX + 1))

    def long_miss_score(n: int) -> float:
        return stats.blue_counts[n] * 3.0 + min(stats.blue_miss[n], 20) * 0.7

    def warm_score(n: int) -> float:
        warm_bonus = 2.0 if stats.blue_miss[n] <= 6 else 0.0
        recent_curve = max(0, 8 - stats.blue_miss[n]) * 0.4
        return stats.blue_counts[n] * 3.5 + warm_bonus + recent_curve

    by_long_miss = sorted(items, key=lambda n: (long_miss_score(n), -n), reverse=True)
    by_warm = sorted(items, key=lambda n: (warm_score(n), -n), reverse=True)
    by_recent = sorted(items, key=lambda n: (stats.blue_miss[n], -stats.blue_counts[n], n))
    return sorted(unique_take([*by_long_miss[:3], *by_warm[:3], *by_recent[:3], *by_long_miss], size))


def unique_take(items: list[int], size: int) -> list[int]:
    picked: list[int] = []
    for item in items:
        if item not in picked:
            picked.append(item)
        if len(picked) >= size:
            break
    return picked


def choose_dan(red_candidates: list[int], stats: Stats, count: int = 1) -> list[int]:
    scored = []
    for n in red_candidates:
        count_score = stats.red_counts[n] * 2.8
        miss_score = min(stats.red_miss[n], 20) * 0.6
        stability = 1.0 if 0 < stats.red_counts[n] <= 4 else 0.0
        warm_miss = 0.8 if 2 <= stats.red_miss[n] <= 6 else 0.0
        scored.append((count_score + miss_score + stability + warm_miss, n))
    scored.sort(reverse=True)

    picked: list[int] = []
    seen_regions = set()
    for _, n in scored:
        if len(picked) >= count:
            break
        picked.append(n)
        seen_regions.add(region(n))
    return sorted(picked)


def combo_score(combo: tuple[int, ...], stats: Stats) -> tuple[float, dict[str, float]]:
    red = tuple(sorted(combo))
    odd = sum(1 for n in red if n % 2 == 1)
    small = sum(1 for n in red if n <= 16)
    regions = [0, 0, 0]
    for n in red:
        regions[region(n)] += 1
    repeats = len(set(red) & stats.last_red)
    consecutive = sum(1 for a, b in zip(red, red[1:]) if b == a + 1)
    total = sum(red)

    heat_avg = statistics.mean(stats.red_counts[n] for n in red)
    miss_avg = statistics.mean(stats.red_miss[n] for n in red)
    high_region_count = sum(1 for n in red if region(n) == 2)
    middle_region_count = sum(1 for n in red if region(n) == 1)
    high_a = sum(1 for n in red if high_band(n) == 1)
    high_b = sum(1 for n in red if high_band(n) == 2)
    push_mode = stats.high_streak >= 2
    target_sum = stats.sum_mean + stats.sum_std * 0.8 if push_mode else stats.sum_mean

    heat_score = min(15.0, heat_avg * 3.0)
    miss_score = min(15.0, max(0.0, miss_avg * 1.2))

    if all(x > 0 for x in regions):
        region_score = 15.0
    elif middle_region_count > 0 and sum(x > 0 for x in regions) == 2:
        region_score = 12.0
    elif middle_region_count > 0:
        region_score = 7.5
    else:
        region_score = 2.25

    odd_score = score_balance(odd, target=(3, 3))
    size_score = score_balance(small, target=(3, 3), small_count=True)

    sum_score = score_sum(total, target_sum, stats.sum_std)
    consecutive_score = 10.0 if consecutive <= 1 else 4.0 if consecutive == 2 else 0.0
    repeat_score = 5.0 if repeats <= 2 else 1.0
    high_score = (
        min(3.0, high_b * 0.9) + min(0.2, high_a * 0.2)
        if push_mode
        else min(3.0, high_a * 0.9) + min(0.1, high_b * 0.1)
    )

    total_score = (
        heat_score
        + miss_score
        + region_score
        + odd_score
        + size_score
        + sum_score
        + consecutive_score
        + repeat_score
        + high_score
    )

    parts = {
        "heat": heat_score,
        "miss": miss_score,
        "region": region_score,
        "odd": odd_score,
        "size": size_score,
        "sum": sum_score,
        "consecutive": consecutive_score,
        "repeat": repeat_score,
        "high": high_score,
    }
    return total_score, parts


def score_balance(value: int, *, target: tuple[int, int], small_count: bool = False) -> float:
    a, b = target
    if value == a:
        return 15.0
    if value in (a - 1, a + 1):
        return 12.0
    if value in (a - 2, a + 2):
        return 8.0
    return 3.0


def score_sum(total: int, mean: float, std: float) -> float:
    if std == 0:
        return 10.0
    lo = mean - std
    hi = mean + std
    if lo <= total <= hi:
        center = mean
        dist = abs(total - center)
        return 10.0 if dist <= std * 0.25 else 8.0 if dist <= std * 0.5 else 6.0
    return 2.0 if abs(total - mean) <= std * 1.5 else 0.0


def valid_combo(combo: tuple[int, ...], stats: Stats) -> bool:
    red = tuple(sorted(combo))
    odd = sum(1 for n in red if n % 2 == 1)
    small = sum(1 for n in red if n <= 16)
    regions = {region(n) for n in red}
    middle_count = sum(1 for n in red if region(n) == 1)
    high_a = sum(1 for n in red if high_band(n) == 1)
    high_b = sum(1 for n in red if high_band(n) == 2)
    total = sum(red)
    repeats = len(set(red) & stats.last_red)
    consecutive = sum(1 for a, b in zip(red, red[1:]) if b == a + 1)
    push_mode = stats.high_streak >= 2

    if odd not in {2, 3, 4}:
        return False
    if small not in {2, 3, 4}:
        return False
    if len(regions) < 2:
        return False
    if middle_count < 1:
        return False
    if consecutive > 2:
        return False
    if repeats > 3:
        return False
    if push_mode and high_b < 1:
        return False
    if not push_mode and high_a < 1:
        return False
    if stats.sum_std > 0 and not (
        (stats.sum_mean - 0.5 * stats.sum_std <= total <= stats.sum_mean + 3.0 * stats.sum_std)
        if push_mode
        else (stats.sum_mean - 3.0 * stats.sum_std <= total <= stats.sum_mean + 2.5 * stats.sum_std)
    ):
        return False
    return True


def blue_score(n: int, stats: Stats) -> float:
    heat = stats.blue_counts[n] * 2.5
    miss = min(stats.blue_miss[n], 20) * 0.8
    return heat + miss


def format_combo(red: tuple[int, ...], blue: int) -> str:
    red_str = " ".join(f"{n:02d}" for n in red)
    return f"{red_str} + {blue:02d}"


def generate(draws: list[Draw], top_n: int = 5) -> list[dict]:
    stats = calc_stats(draws)
    red_candidates = red_pool(stats)
    blues = blue_pool(stats)
    dans = choose_dan(red_candidates, stats)
    push_mode = stats.high_streak >= 2

    results = []
    balanced_combos = [combo for combo in combinations(red_candidates, 6) if set(dans).issubset(combo) and valid_combo(combo, stats)]
    if push_mode:
        extra_combos = [
            combo for combo in combinations(red_candidates, 6)
            if sum(1 for n in combo if high_band(n) == 2) >= 1
            and sum(1 for n in combo if high_band(n) == 1) >= 1
            and sum(combo) >= stats.sum_mean + stats.sum_std * 0.25
            and any(n >= 23 for n in combo)
        ]
    else:
        extra_combos = [
            combo for combo in combinations(red_candidates, 6)
            if sum(1 for n in combo if high_band(n) == 1) >= 2
            and sum(combo) <= stats.sum_mean + stats.sum_std * 0.8
            and any(n >= 23 for n in combo)
        ]

    for combo in balanced_combos + extra_combos:
        total_score, parts = combo_score(combo, stats)
        for blue in blues:
            final_score = total_score + blue_score(blue, stats)
            results.append(
                {
                    "red": tuple(sorted(combo)),
                    "blue": blue,
                    "score": round(final_score, 2),
                    "parts": {k: round(v, 2) for k, v in parts.items()},
                    "text": format_combo(tuple(sorted(combo)), blue),
                    "mode": "push" if push_mode and sum(1 for n in combo if high_band(n) == 2) >= 1 else "pull",
                }
            )

    results.sort(key=lambda x: (x["score"], x["text"]), reverse=True)
    return diversify_results(results, top_n)


def diversify_results(results: list[dict], top_n: int) -> list[dict]:
    deduped = []
    seen = set()
    for item in results:
        key = (item["red"], item["blue"])
        if key in seen:
            continue
        if any(overlap_count(picked["red"], item["red"]) >= 5 for picked in deduped):
            continue
        if any(overlap_count(picked["red"], item["red"]) >= 4 and picked["blue"] == item["blue"] for picked in deduped):
            continue
        if sum(1 for picked in deduped if picked["blue"] == item["blue"]) >= 2:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_n:
            break
    if len(deduped) < top_n:
        for item in results:
            key = (item["red"], item["blue"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= top_n:
                break
    return deduped


def overlap_count(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    b_set = set(b)
    return sum(1 for n in a if n in b_set)


def load_sample() -> list[Draw]:
    raw = [
        {"period": "2026001", "red": [1, 5, 12, 18, 24, 31], "blue": 9},
        {"period": "2026002", "red": [3, 8, 11, 16, 22, 30], "blue": 14},
        {"period": "2026003", "red": [2, 6, 13, 17, 25, 33], "blue": 2},
        {"period": "2026004", "red": [4, 7, 14, 19, 23, 29], "blue": 11},
        {"period": "2026005", "red": [1, 9, 10, 21, 26, 32], "blue": 16},
        {"period": "2026006", "red": [5, 6, 15, 20, 27, 30], "blue": 7},
        {"period": "2026007", "red": [2, 8, 12, 18, 24, 28], "blue": 4},
        {"period": "2026008", "red": [3, 7, 11, 17, 22, 31], "blue": 10},
        {"period": "2026009", "red": [4, 9, 13, 19, 25, 33], "blue": 1},
        {"period": "2026010", "red": [1, 6, 14, 16, 23, 29], "blue": 12},
        {"period": "2026011", "red": [5, 8, 10, 18, 27, 32], "blue": 6},
        {"period": "2026012", "red": [2, 7, 15, 20, 26, 30], "blue": 15},
        {"period": "2026013", "red": [3, 9, 12, 21, 24, 31], "blue": 3},
        {"period": "2026014", "red": [4, 6, 11, 17, 28, 33], "blue": 8},
        {"period": "2026015", "red": [1, 8, 13, 19, 22, 29], "blue": 13},
        {"period": "2026016", "red": [2, 5, 14, 18, 25, 30], "blue": 5},
        {"period": "2026017", "red": [3, 7, 10, 16, 23, 32], "blue": 2},
        {"period": "2026018", "red": [4, 9, 15, 20, 27, 31], "blue": 9},
        {"period": "2026019", "red": [1, 6, 12, 17, 24, 33], "blue": 14},
        {"period": "2026020", "red": [2, 8, 11, 18, 26, 29], "blue": 7},
    ]
    return [Draw(period=item["period"], red=tuple(item["red"]), blue=item["blue"]) for item in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="双色球选号候选生成器")
    parser.add_argument("--input", type=Path, help="输入 JSON 文件，格式为 [{period, red, blue}, ...]")
    parser.add_argument("--top", type=int, default=5, help="输出前几个候选")
    parser.add_argument("--sample", action="store_true", help="使用内置示例数据")
    args = parser.parse_args()

    if args.sample or not args.input:
        draws = load_sample()
    else:
        draws = parse_draws(args.input)

    if len(draws) < 10:
        raise SystemExit("至少需要 10 期数据，建议输入近 20 期。")

    results = generate(draws, top_n=args.top)
    if not results:
        raise SystemExit("没有生成符合条件的候选号码，请放宽筛选规则。")

    print(f"输入期数: {len(draws)}")
    print("候选结果:")
    for idx, item in enumerate(results, start=1):
        p = item["parts"]
        print(
            f"{idx}. {item['text']} | 分数 {item['score']} | "
            f"热度 {p['heat']} 遗漏 {p['miss']} 区间 {p['region']} 奇偶 {p['odd']} 大小 {p['size']} "
            f"和值 {p['sum']} 连号 {p['consecutive']} 重号 {p['repeat']}"
        )


if __name__ == "__main__":
    main()
