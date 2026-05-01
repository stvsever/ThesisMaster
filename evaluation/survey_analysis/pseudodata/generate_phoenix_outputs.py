"""
Generate pseudo PHOENIX outputs for every case.

Same canonical shape as the HCP pseudodata but with distinct phrasing so the
LLM judge (or pseudo judge) sees genuinely different content. The judge is
blind to the source by construction; the textual differences here matter
only when a real judge is wired in.
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
    coerce_part4,
)

_CASES: Dict[str, Dict[str, Any]] = {
    "C01": {
        "part1": [
            ("anhedonia", "diminished interest or pleasure across most activities"),
            ("insomnia (initial)", "prolonged sleep latency, often >30 min"),
            ("psychomotor slowing", "subjective slowness in everyday tasks"),
            ("rumination", "repetitive negative self-focused cognition"),
            ("interpersonal withdrawal", "reduced reciprocal contact with peers"),
            ("low affective baseline", "morning mood consistently below 4/7"),
        ],
        "part2": [
            ("sleep latency", "self-reported minutes-to-sleep", "<30 min target"),
            ("morning affect", "single-item Likert on wake", ">=4 on 1..7"),
            ("daily activity", "step count via phone", ">=4500/day"),
            ("rumination intensity", "Likert prompt", "<=3 on 1..7"),
            ("social interaction count", "tally per evening prompt", ">=1/day"),
        ],
        "part3_ranking": [2, 1, 4, 3, 5],
        "part4": "rumination intensity, morning affect, social interaction count //note: drop activity, near-saturation",
        "part5": (
            "Showing up consistently for monitoring is itself progress, especially on heavier days. "
            "For tomorrow morning, schedule one 12-minute walk before noon and we'll see how mood shifts."
        ),
    },
    "C02": {
        "part1": [
            ("anticipatory anxiety", "rising worry preceding social-evaluative situations"),
            ("safety-behaviour use", "rehearsing answers, avoiding eye contact"),
            ("autonomic activation", "heart-rate spikes, jaw tension"),
            ("cognitive distortions", "mind-reading and catastrophising about peers"),
        ],
        "part2": [
            ("anticipatory worry", "Likert pre-meeting prompt", "<=3 on 1..7"),
            ("safety-behaviour count", "self-report tally per situation", "<=1 per meeting"),
            ("perceived control", "single-item Likert", ">=4 on 1..7"),
            ("body-tension rating", "Likert", "<=3 on 1..7"),
            ("post-event self-criticism", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [1, 2, 4, 3, 5],
        "part4": "anticipatory worry, safety-behaviour count, post-event self-criticism",
        "part5": (
            "Stepping into a feared situation while still anxious is the change-mechanism, not bravery. "
            "Tomorrow, enter the meeting one minute early and log control rating immediately after."
        ),
    },
    "C03": {
        "part1": [
            ("low motivation", "reduced goal-directed initiation"),
            ("anhedonia", "blunted hedonic response"),
            ("cognitive fatigue", "subjective concentration difficulty"),
            ("interpersonal pull-back", "withdrawal from non-essential contacts"),
            ("sleep fragmentation", "frequent night awakenings"),
        ],
        "part2": [
            ("sleep fragmentation", "self-rated number of awakenings", "<=2/night"),
            ("morning energy", "single-item Likert", ">=4 on 1..7"),
            ("activity initiation count", "tally", ">=1 task initiated"),
            ("recovery activity", "minutes/day", ">=20 min"),
        ],
        "part3_ranking": [1, 4, 3, 5, 2],
        "part4": "sleep fragmentation, morning energy, recovery activity, work-thoughts shutdown ritual //note: tighten evening prompts",
        "part5": (
            "You already noticed the screen-sleep loop, that's clinical insight worth using. "
            "For tonight: phone in another room from 21:30 and one line in the recovery log before sleep."
        ),
    },
    "C04": {
        "part1": [
            ("re-experiencing", "intrusive sensory memories with distress"),
            ("emotional blunting", "reduced affective range with intimates"),
            ("hyper-arousal", "exaggerated startle and vigilance"),
            ("avoidance pattern", "behavioural avoidance of trauma cues"),
            ("sleep disturbance", "vivid dreams disrupting continuity"),
        ],
        "part2": [
            ("intrusion frequency", "tally per day prompt", "<=2/day"),
            ("emotional connection", "Likert with partner", ">=4 on 1..7"),
            ("startle reactivity", "Likert at evening prompt", "<=3 on 1..7"),
            ("avoidance episodes", "tally", "<=1/day"),
            ("nightmare incidence", "yes/no on wake", "downward trend"),
        ],
        "part3_ranking": [1, 3, 2, 4, 5],
        "part4": "intrusion frequency, emotional connection, nightmare incidence",
        "part5": (
            "Your nervous system is treating yesterday like today; that takes energy. "
            "Tomorrow, sit at the cafe an extra five minutes and log the body-tension rating before leaving."
        ),
    },
    "C05": {
        "part1": [
            ("loss-of-control eating", "subjective inability to stop eating"),
            ("body-image distress", "negative thoughts about body following meals"),
            ("emotion-driven intake", "eating triggered by stress or boredom"),
            ("dietary restraint cycling", "alternating restriction and binge"),
        ],
        "part2": [
            ("hunger pre-meal", "Likert", "moderate range"),
            ("emotion pre-snack", "label + intensity", "log all"),
            ("urge intensity", "Likert", "<=3 on 1..7"),
            ("mindful pause used", "yes/no", ">=1/urge"),
            ("body-image self-talk", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [1, 2, 4, 3, 5],
        "part4": "urge intensity, mindful pause used, emotion pre-snack //note: keep restraint cycling watch",
        "part5": (
            "Catching the urge before acting on it is the leverage point. "
            "Next urge: 5-minute timer, write one sentence about the feeling, then decide."
        ),
    },
    "C06": {
        "part1": [
            ("evening craving", "alcohol urge in late afternoon and evening"),
            ("stress-driven trigger", "end-of-workday tension precipitating drinking"),
            ("morning regret", "post-drinking dysphoria and self-criticism"),
            ("sleep architecture loss", "fragmented and non-restorative sleep on drinking nights"),
        ],
        "part2": [
            ("craving intensity", "Likert", "<=3 on 1..7"),
            ("evening stress", "Likert", "tracked daily"),
            ("alternative-coping use", "tally", ">=1/day"),
            ("morning regret", "Likert", "<=3 on 1..7"),
            ("drink-free streak", "days since last drink", "incrementing"),
        ],
        "part3_ranking": [1, 2, 4, 5, 3],
        "part4": "craving intensity, evening stress, alternative-coping use, drink-free streak",
        "part5": (
            "Each evening that ends without a drink is a real data point worth banking. "
            "Tonight, schedule one alternative coping action at 18:00 and tick it off when done."
        ),
    },
    "C07": {
        "part1": [
            ("panic episodes", "abrupt surges of fear with somatic symptoms"),
            ("anticipatory anxiety", "worry about subsequent attacks"),
            ("interoceptive sensitivity", "heightened attention to body signals"),
            ("agoraphobic avoidance", "avoiding public transport and shops"),
        ],
        "part2": [
            ("panic count", "tally/day", "<=1/day"),
            ("avoidance episodes", "tally", "<=1/day"),
            ("interoceptive monitoring", "Likert", "tracked"),
            ("breath-pacing practice", "minutes/day", ">=5 min"),
            ("anticipatory worry", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [1, 3, 2, 5, 4],
        "part4": "panic count, breath-pacing practice, anticipatory worry",
        "part5": (
            "Your alarm system is firing on safe stimuli; we can teach it new patterns. "
            "Try a 4-minute paced-breath set tomorrow morning before the commute."
        ),
    },
    "C08": {
        "part1": [
            ("intrusive obsessions", "recurrent, ego-dystonic doubts about contamination"),
            ("checking compulsions", "repetitive behaviours to neutralise anxiety"),
            ("ERP avoidance", "putting off planned exposure exercises"),
            ("functional disruption", "missed work hours due to rituals"),
        ],
        "part2": [
            ("obsession intensity", "Likert at prompt", "tracked"),
            ("compulsion duration", "minutes/day", "<=30 min"),
            ("ERP exposures", "count of completed exercises", ">=1/day"),
            ("post-ERP anxiety drop", "Likert pre/post", "post < pre"),
            ("functional impact", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [2, 1, 3, 5, 4],
        "part4": "ERP exposures, post-ERP anxiety drop, compulsion duration",
        "part5": (
            "Allowing the urge without performing the ritual is the active ingredient. "
            "Today: one ERP from the hierarchy and log how the urge declines after five minutes."
        ),
    },
    "C09": {
        "part1": [
            ("hopelessness", "pessimistic future outlook"),
            ("passive suicidal ideation", "wish not to wake up without intent"),
            ("interpersonal isolation", "reduced contact with support figures"),
            ("anhedonia", "diminished reward responsiveness"),
            ("appetite reduction", "decreased intake"),
        ],
        "part2": [
            ("hopelessness rating", "Likert", "tracked closely"),
            ("connection events", "tally", ">=1/day"),
            ("safety-plan use", "yes/no", "active"),
            ("appetite", "self-report", ">=2 meals"),
            ("ideation intensity", "Likert", "<=3 on 1..7"),
        ],
        "part3_ranking": [1, 3, 2, 4, 5],
        "part4": "hopelessness rating, connection events, safety-plan use, ideation intensity //note: clinician check-in same day if Likert >=5",
        "part5": (
            "When the weight is heavy, micro-connections still register. "
            "Today, message one person on your safety list and log how connected you felt afterward."
        ),
    },
    "C10": {
        "part1": [
            ("emotional exhaustion", "chronic energy depletion at end of workday"),
            ("depersonalisation at work", "detachment from work-relationships"),
            ("efficacy doubt", "reduced confidence in own competence"),
            ("non-restorative sleep", "wake unrefreshed despite sleep duration"),
        ],
        "part2": [
            ("end-of-day exhaustion", "Likert", "<=4 on 1..7"),
            ("meaning-of-work rating", "Likert", ">=4 on 1..7"),
            ("recovery activities", "tally", ">=1/day"),
            ("micro-breaks", "tally during workday", ">=3/day"),
            ("morning restoration", "Likert at wake", ">=4 on 1..7"),
        ],
        "part3_ranking": [1, 4, 2, 3, 5],
        "part4": "end-of-day exhaustion, micro-breaks, recovery activities, morning restoration",
        "part5": (
            "Running on empty is signal, not failure; rest is part of the job. "
            "Tomorrow, block 15 minutes for a walk before lunch and protect it like any meeting."
        ),
    },
}


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
    p4 = coerce_part4(case_payload["part4"])
    p5 = Part5Output(message=case_payload["part5"])
    return {
        "part1": p1.to_dict(),
        "part2": p2.to_dict(),
        "part3": p3.to_dict(),
        "part4": p4.to_dict(),
        "part5": p5.to_dict(),
    }


def generate_phoenix_outputs(out_path: Path | None = None) -> Path:
    """Build pseudo PHOENIX outputs and write them to disk."""
    out_path = Path(out_path) if out_path else PSEUDODATA_DIR / "phoenix_outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {case_id: _case_to_canonical(payload)
              for case_id, payload in _CASES.items()}
    out_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    p = generate_phoenix_outputs()
    print(f"Wrote pseudo PHOENIX outputs to {p}")
