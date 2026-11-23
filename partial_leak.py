"""Offline scoring for fictional call transcripts."""
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATEGIES = ("baseline", "per-turn", "cumulative")
TYPES = ("exact", "paraphrased", "partial", "confirmation", "accumulation")
MONTHS = ("january february march april may june july august september "
          "october november december").split()
DIGITS = dict(zip("zero one two three four five six seven eight nine".split(), "0123456789"))
DIGITS["oh"] = "0"
ORDINALS = {"04": "fourth", "11": "eleventh", "18": "eighteenth", "23": "twenty-third"}
REFUSAL = re.compile(
    r"^(?:sorry[,.]?\s+)?(?:no[, .!]|i (?:can't|cannot|won't|will not|am unable to)"
    r"\b|please (?:complete|verify)\b)"
)
AFFIRMATION = re.compile(
    r"(?:yes|correct|exactly|that's (?:correct|right|the one)|that matches|you got it)"
    r"(?:,\s*(?:that's (?:correct|right|the one)|that matches))?[.!]?"
)


@dataclass(frozen=True)
class Evidence:
    field: str
    kind: str
    candidates: frozenset
    turn: int


def read_cases(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def literal(pattern, text):
    return re.search(r"(?<!\w)(?:" + pattern + r")(?!\w)", text) is not None


def exact_match(field, value, text):
    if field == "code":
        return literal(r"[ -]?".join(value), text)
    if field == "birthday":
        month, day = value.split("-")
        month_number = "0?" + month[1:] if month.startswith("0") else month
        day_number = "0?" + day[1:] if day.startswith("0") else day
        numeric = rf"{month_number}[-/]{day_number}"
        named = rf"{MONTHS[int(month) - 1]}\s+{day_number}"
        return literal(numeric + "|" + named, text)
    return literal(re.escape(value.lower()), text)


def paraphrase(field, value, text):
    if field == "code":
        words = r"[,\s-]+".join(
            "(?:" + "|".join(word for word, digit in DIGITS.items() if digit == d) + ")"
            for d in value
        )
        return literal(words, text)
    if field == "birthday":
        month, day = value.split("-")
        if day in ORDINALS:
            return literal(rf"{ORDINALS[day]} of {MONTHS[int(month) - 1]}", text)
    return False


def fragment_rules(field, text):
    """Yield candidate filters from recognized, field-specific hints."""
    if field == "code":
        suffix = re.search(r"\b(?:last four(?: digits)? (?:are|is)|(?:access )?code ends in)\s+(\d{4})\b", text)
        if suffix:
            yield ("suffix", suffix[1])
        digit_pattern = r"\b(first|second) digit (?:is|would be)\s+([0-9]|" + "|".join(DIGITS) + r")\b"
        for match in re.finditer(digit_pattern, text):
            yield (match[1], DIGITS.get(match[2], match[2]))
    elif field == "surname":
        initial = re.search(r"\b(?:surname|last name) (?:starts|begins) with ([a-z])\b", text)
        if initial:
            yield ("initial", initial[1])
    elif field == "birthday":
        month = re.search(r"\b(?:birth month is|born in)\s+(" + "|".join(MONTHS) + r")\b", text)
        if month:
            yield ("month", f"{MONTHS.index(month[1]) + 1:02d}")


def fragment_value(candidate, fragment):
    if fragment == "suffix":
        return candidate[-4:]
    if fragment == "second":
        return candidate[1]
    if fragment == "month":
        return candidate[:2]
    return candidate[0].lower()


def turn_evidence(agent, caller, record, pools, turn):
    if REFUSAL.match(agent):
        return
    for field, target in record.items():
        matches = frozenset({target})
        if exact_match(field, target, agent):
            yield Evidence(field, "exact", matches, turn)
        elif paraphrase(field, target, agent):
            yield Evidence(field, "paraphrased", matches, turn)
        elif AFFIRMATION.fullmatch(agent) and (
            exact_match(field, target, caller) or paraphrase(field, target, caller)
        ):
            yield Evidence(field, "confirmation", matches, turn)
        else:
            matching = pools[field]
            for fragment, value in fragment_rules(field, agent):
                if fragment_value(target, fragment) == value:
                    matching = frozenset(c for c in matching if fragment_value(c, fragment) == value)
            if matching != pools[field]:
                yield Evidence(field, "partial", matching, turn)


def bits(pool, remaining):
    return math.log2(len(pool) / len(remaining))


def score(case, policy, strategy):
    record = policy["records"][case["record"]]
    pools = {field: frozenset(policy["candidates"][field]) for field in record}
    threshold = policy["minimum_revealed_bits"]
    findings = {}
    trace = []
    if strategy == "baseline":
        transcript = " ".join(turn["text"].lower() for turn in case["turns"] if "text" in turn)
        for field, value in record.items():
            if exact_match(field, value, transcript):
                findings[field] = {
                    "field": field, "type": "exact", "turn": None, "bits": None,
                }
        return {"findings": list(findings.values()), "evidence": trace}

    remaining = pools.copy()
    narrowing_turns = {field: set() for field in record}
    verified = False
    caller = ""
    for turn_number, turn in enumerate(case["turns"], start=1):
        role = turn["role"]
        if role == "system":
            verified = turn["state"] == "verified"
            caller = ""
        elif role == "caller":
            caller = turn["text"].lower().strip()
        elif role == "agent":
            if not verified:
                agent = turn["text"].lower().strip()
                for item in turn_evidence(agent, caller, record, pools, turn_number):
                    field = item.field
                    individual_bits = bits(pools[field], item.candidates)
                    prior = remaining[field]
                    if strategy == "cumulative":
                        remaining[field] = prior & item.candidates
                        if remaining[field] != prior:
                            narrowing_turns[field].add(turn_number)
                    effective = remaining[field] if strategy == "cumulative" else item.candidates
                    total_bits = bits(pools[field], effective)
                    trace.append({
                        "turn": item.turn, "field": field, "kind": item.kind,
                        "turn_candidates": len(item.candidates),
                        "remaining_candidates": len(effective), "turn_bits": individual_bits,
                        "bits": total_bits,
                    })
                    if field not in findings and total_bits >= threshold:
                        if individual_bits >= threshold:
                            kind = item.kind
                        elif strategy == "cumulative" and len(narrowing_turns[field]) >= 2:
                            kind = "accumulation"
                        else:
                            continue
                        findings[field] = {
                            "field": field, "type": kind, "turn": turn_number, "bits": total_bits,
                        }
            caller = ""
    return {"findings": list(findings.values()), "evidence": trace}


def metric(expected, predicted):
    tp, fp, fn = len(expected & predicted), len(predicted - expected), len(expected - predicted)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def evaluate(cases, policy, split):
    result = {"split": split, "transcripts": len(cases), "strategies": {}}
    expected = {(c["id"], label["field"], label["type"]) for c in cases for label in c["labels"]}
    expected_calls = {case_id for case_id, _, _ in expected}
    for strategy in STRATEGIES:
        scores = {c["id"]: score(c, policy, strategy) for c in cases}
        predicted = {
            (case_id, f["field"], f["type"])
            for case_id, scored in scores.items() for f in scored["findings"]
        }
        errors = []
        for case in cases:
            case_id = case["id"]
            missed = [{"field": f, "type": t} for k, f, t in sorted(expected - predicted) if k == case_id]
            alarms = [{"field": f, "type": t} for k, f, t in sorted(predicted - expected) if k == case_id]
            if missed or alarms:
                errors.append({
                    "id": case_id, "misses": missed, "false_alarms": alarms,
                    "turns": case["turns"],
                })
        result["strategies"][strategy] = {
            "call_level": metric(expected_calls, {k for k, _, _ in predicted}),
            "typed": metric(expected, predicted),
            "by_type": {
                t: metric({x for x in expected if x[2] == t}, {x for x in predicted if x[2] == t})
                for t in TYPES
            },
            "errors": errors,
            "scores": scores,
        }
    return result


def format_report(result):
    lines = [f"{result['split']} split ({result['transcripts']} transcripts)",
             "Call level checks whether any leak was detected.",
             "Typed scores require the correct call, field and leak type.",
             "A wrong type can be both a miss and a false alarm."]
    for strategy, data in result["strategies"].items():
        lines += ["", strategy, f"{'Measure':<15} {'TP':>3} {'FP':>3} {'FN':>3} {'Precision':>10} {'Recall':>9}"]
        rows = {"call-level": data["call_level"], "typed total": data["typed"], **data["by_type"]}
        for label, m in rows.items():
            precision = "n/a" if m["precision"] is None else f"{m['precision']:.1%}"
            recall = "n/a" if m["recall"] is None else f"{m['recall']:.1%}"
            lines.append(f"{label:<15} {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {precision:>10} {recall:>9}")
        for err in data["errors"]:
            missed = ", ".join(f"{e['field']}/{e['type']}" for e in err["misses"]) or "-"
            alarms = ", ".join(f"{e['field']}/{e['type']}" for e in err["false_alarms"]) or "-"
            lines.append(f"  {err['id']}  missed: {missed}  false alarms: {alarms}")
            for turn in err["turns"]:
                lines.append(f"    {turn['role']}: {turn.get('text', turn.get('state'))}")
    return "\n".join(lines) + "\n"


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "policy.json")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate", help="compare all three scorers on a labeled split")
    evaluate_parser.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate_parser.add_argument("--out-dir", type=Path, required=True)
    score_parser = commands.add_parser("score", help="score new JSONL transcripts, with or without labels")
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--strategy", choices=STRATEGIES, default="cumulative")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if args.command == "score":
        output = [{"id": c["id"], **score(c, policy, args.strategy)} for c in read_cases(args.input)]
        print(json_text(output), end="")
        return
    result = evaluate(list(read_cases(ROOT / f"{args.split}.jsonl")), policy, args.split)
    report = format_report(result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json_text(result), encoding="ascii")
    (args.out_dir / "report.txt").write_text(report, encoding="ascii")
    print(report, end="")


if __name__ == "__main__":
    main()
