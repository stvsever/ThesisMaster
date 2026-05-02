"""
Generate pseudo HCP outputs covering all 10 cases.

The content does not need to be clinically perfect; it only needs to look
like the kind of structured-but-imperfect text a clinician would produce.
The shape is the canonical per-part dict so the downstream judge can be
exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from analysis.shared.survey_paths import PSEUDODATA_DIR
from parsing.canonical_schemas import (
    Part1Item, Part1Output,
    Part2Item, Part2Output,
    Part3Item, Part3Output,
    Part4Output, Part5Output,
)


# ──────────────────────────────────────────────────────────────────────────────
# Static pseudo content per case
# ──────────────────────────────────────────────────────────────────────────────

_CASES: Dict[str, Dict[str, Any]] = {
    "C01": {
        "part1": [
            ("low energy", "persistent fatigue throughout the day"),
            ("anhedonia", "loss of pleasure in usual activities"),
            ("sleep onset insomnia", "difficulty falling asleep at night"),
            ("ruminative thoughts", "repetitive worries about work performance"),
            ("social withdrawal", "decreased contact with friends and colleagues"),
        ],
        "part2": [
            ("sleep duration", "self-report at wake-up", ">=7 hours target"),
            ("morning mood", "single-item Likert", ">=4 on 1..7"),
            ("activity steps", "phone pedometer", ">=5000/day"),
            ("rumination episodes", "count via prompts", "<=3/day"),
        ],
        "part3_ranking": [3, 1, 5, 2, 4],
        "part4": "sleep duration completed today (yes/no), sleep duration duration today (minutes), morning mood completed today (yes/no), morning mood duration today (minutes), activity steps completed today (yes/no), activity steps duration today (minutes)",
        "part5": (
            "You have been showing up even on the harder days, that takes effort. "
            "Tonight, try one 10-minute walk before dinner; we will check how it lands tomorrow."
        ),
    },
    "C02": {
        "part1": [
            ("anxiety: anticipatory", "worry before social events"),
            ("avoidance behaviour", "skipping team meetings"),
            ("muscle tension", "neck and jaw tension by evening"),
            ("hypervigilance", "scanning for criticism in conversations"),
        ],
        "part2": [
            ("anticipatory worry rating", "EMA prompt", ">=3 on 1..7"),
            ("avoidance count", "self-report tally", "<=1 avoidance/day"),
            ("perceived control", "single-item Likert", ">=4 on 1..7"),
            ("body tension", "self-rating", "<=3 on 1..7"),
        ],
        "part3_ranking": [1, 3, 2, 4, 5],
        "part4": "anticipatory worry rating completed today (yes/no), anticipatory worry rating duration today (minutes), avoidance count completed today (yes/no), avoidance count duration today (minutes), perceived control completed today (yes/no), perceived control duration today (minutes)",
        "part5": (
            "It is understandable to want to step back when meetings feel intense. "
            "Try entering tomorrow's meeting two minutes early and just say hi to one person."
        ),
    },
    "C03": {
        "part1": [
            ("low energy", "persistent fatigue"),
            ("apathy", "low motivation to start tasks"),
            ("social withdrawal", "reduced contact with peers"),
            ("forgetfulness", "missing appointments"),
            ("shame", "self-criticism after mistakes"),
        ],
        "part2": [
            ("sleep quality", "single-item rating", ">=4 on 1..7"),
            ("nutrition adequacy", "meal-tally self-report", "3 meals/day"),
            ("physical movement", "minutes of activity", ">=20 min/day"),
        ],
        "part3_ranking": [2, 4, 3, 5, 1],
        "part4": "sleep quality completed today (yes/no), sleep quality duration today (minutes), nutrition adequacy completed today (yes/no), nutrition adequacy duration today (minutes), physical movement completed today (yes/no), physical movement duration today (minutes)",
        "part5": (
            "You know late screen-time is hurting your sleep, but it still feels like "
            "the easiest way to unwind. Tonight, leave your phone in another room at 21:30 "
            "and write one line in the recovery log."
        ),
    },
    "C04": {
        "part1": [
            ("intrusive memories", "trauma-related flashbacks"),
            ("emotional numbing", "feeling distant from loved ones"),
            ("hyperarousal", "exaggerated startle response"),
            ("avoidance", "avoiding crowded places"),
        ],
        "part2": [
            ("flashback frequency", "count per day", "<=2/day"),
            ("emotional connection", "Likert with partner", ">=4 on 1..7"),
            ("startle reactivity", "self-rated", "<=3 on 1..7"),
            ("avoidance episodes", "tally", "<=1/day"),
        ],
        "part3_ranking": [1, 2, 4, 3, 5],
        "part4": "flashback frequency completed today (yes/no), flashback frequency duration today (minutes), emotional connection completed today (yes/no), emotional connection duration today (minutes), startle reactivity completed today (yes/no), startle reactivity duration today (minutes)",
        "part5": (
            "It takes courage to keep showing up after what you went through. "
            "Tomorrow, try staying five extra minutes at the cafe before leaving."
        ),
    },
    "C05": {
        "part1": [
            ("binge episodes", "loss-of-control eating"),
            ("body shame", "negative self-talk after meals"),
            ("emotional eating triggers", "stress-driven snacking"),
        ],
        "part2": [
            ("hunger before meals", "Likert", "moderate range"),
            ("emotion before snacking", "label + intensity", "log all events"),
            ("urge intensity", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [2, 1, 5, 3, 4],
        "part4": "hunger before meals completed today (yes/no), hunger before meals duration today (minutes), emotion before snacking completed today (yes/no), emotion before snacking duration today (minutes), urge intensity completed today (yes/no), urge intensity duration today (minutes)",
        "part5": (
            "Noticing the urge before acting on it is already real progress. "
            "Next time the urge spikes, set a 5-minute timer and write one sentence about what you feel."
        ),
    },
    "C06": {
        "part1": [
            ("alcohol craving", "desire to drink in the evening"),
            ("relapse triggers", "stress at end of workday"),
            ("guilt", "morning-after self-criticism"),
            ("sleep disruption", "fragmented sleep on drinking nights"),
        ],
        "part2": [
            ("craving intensity", "Likert", "<=3 on 1..7"),
            ("evening stress level", "Likert", "tracked daily"),
            ("alternative-coping use", "count", ">=1/day"),
            ("morning regret", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [1, 3, 2, 5, 4],
        "part4": "craving intensity completed today (yes/no), craving intensity duration today (minutes), evening stress level completed today (yes/no), evening stress level duration today (minutes), alternative-coping use completed today (yes/no), alternative-coping use duration today (minutes)",
        "part5": (
            "Each evening you make it through is worth tracking. "
            "Tonight, plan one alternative coping action for 18:00 and tick it off when done."
        ),
    },
    "C07": {
        "part1": [
            ("panic attacks", "sudden surges of fear with somatic symptoms"),
            ("anticipatory anxiety", "worry about next attack"),
            ("agoraphobic avoidance", "avoiding shops and public transport"),
        ],
        "part2": [
            ("panic count", "tally per day", "<=1/day"),
            ("avoidance episodes", "tally", "<=1/day"),
            ("interoceptive awareness", "Likert", "moderate range"),
            ("breath-control practice", "minutes/day", ">=5 min"),
        ],
        "part3_ranking": [1, 2, 4, 5, 3],
        "part4": "panic count completed today (yes/no), panic count duration today (minutes), avoidance episodes completed today (yes/no), avoidance episodes duration today (minutes), interoceptive awareness completed today (yes/no), interoceptive awareness duration today (minutes)",
        "part5": (
            "Your body is misreading safety signals; that is exhausting. "
            "Tomorrow morning, try a 4-minute slow breath sequence before the commute."
        ),
    },
    "C08": {
        "part1": [
            ("obsessions", "intrusive doubts about contamination"),
            ("compulsions", "repetitive checking and washing"),
            ("functional impairment", "missing work due to rituals"),
        ],
        "part2": [
            ("obsession intensity", "Likert", "tracked"),
            ("compulsion duration", "minutes/day", "<=30 min"),
            ("ERP exposure exercises", "count", ">=1/day"),
            ("anxiety after ERP", "Likert pre/post", "post < pre"),
        ],
        "part3_ranking": [3, 2, 1, 5, 4],
        "part4": "obsession intensity completed today (yes/no), obsession intensity duration today (minutes), compulsion duration completed today (yes/no), compulsion duration duration today (minutes), ERP exposure exercises completed today (yes/no), ERP exposure exercises duration today (minutes)",
        "part5": (
            "Sitting with the discomfort even briefly is the work. "
            "Today, try one ERP from your hierarchy and log how the urge fades after five minutes."
        ),
    },
    "C09": {
        "part1": [
            ("hopelessness", "pessimistic future outlook"),
            ("suicidal ideation passive", "wish not to wake up"),
            ("isolation", "reduced contact with support network"),
            ("appetite changes", "reduced food intake"),
        ],
        "part2": [
            ("hopelessness rating", "Likert", "tracked closely"),
            ("connection events", "tally", ">=1/day"),
            ("safety-plan use", "yes/no", "active"),
            ("food intake", "meals/day", ">=2"),
        ],
        "part3_ranking": [1, 2, 5, 3, 4],
        "part4": "hopelessness rating completed today (yes/no), hopelessness rating duration today (minutes), connection events completed today (yes/no), connection events duration today (minutes), safety-plan use completed today (yes/no), safety-plan use duration today (minutes)",
        "part5": (
            "When everything feels heavy, even one phone call counts. "
            "Today, message one person on your list and rate how connected you felt afterward."
        ),
    },
    "C10": {
        "part1": [
            ("burnout: exhaustion", "chronic depletion at end of day"),
            ("cynicism", "detachment from work meaning"),
            ("efficacy loss", "doubt in own competence"),
            ("sleep impact", "non-restorative sleep"),
        ],
        "part2": [
            ("end-of-day exhaustion", "Likert", "<=4 on 1..7"),
            ("meaning-of-work rating", "Likert", ">=4 on 1..7"),
            ("recovery activities", "count", ">=1/day"),
            ("micro-breaks", "count during workday", ">=3/day"),
        ],
        "part3_ranking": [2, 5, 1, 4, 3],
        "part4": "end-of-day exhaustion completed today (yes/no), end-of-day exhaustion duration today (minutes), meaning-of-work rating completed today (yes/no), meaning-of-work rating duration today (minutes), recovery activities completed today (yes/no), recovery activities duration today (minutes)",
        "part5": (
            "You have been running on empty; that is data, not failure. "
            "Block 15 minutes for a walk before lunch tomorrow and protect it like a meeting."
        ),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────

def _case_to_canonical(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    p1 = Part1Output(items=[Part1Item(label=l)
                            for l, _d in case_payload["part1"]])
    p2 = Part2Output(items=[Part2Item(label=p)
                            for p, _m, _c in case_payload["part2"]])
    ranks = case_payload["part3_ranking"]
    p3 = Part3Output(ranking=sorted(
        [Part3Item(rank=int(r), option_id=f"BO-{i+1}")
         for i, r in enumerate(ranks)],
        key=lambda x: x.rank,
    ))
    # Reuse Part4 coercion pathway by passing the raw string directly.
    from parsing.canonical_schemas import coerce_part4
    p4 = coerce_part4(case_payload["part4"])
    p5 = Part5Output(message=case_payload["part5"])
    return {
        "part1": p1.to_dict(),
        "part2": p2.to_dict(),
        "part3": p3.to_dict(),
        "part4": p4.to_dict(),
        "part5": p5.to_dict(),
    }


def generate_hcp_outputs(out_path: Path | None = None) -> Path:
    """
    Build pseudo HCP outputs and write them to disk.

    Returns the path of the written JSON file.
    """
    out_path = Path(out_path) if out_path else PSEUDODATA_DIR / "hcp_outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {case_id: _case_to_canonical(payload)
              for case_id, payload in _CASES.items()}
    out_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    p = generate_hcp_outputs()
    print(f"Wrote pseudo HCP outputs to {p}")
