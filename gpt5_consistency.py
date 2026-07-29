# pip install openai
#
# Prompts GPT-5 *directly at the occupation level* (no task-level decomposition)
# for all O*NET computer occupations you listed.
#
# Output: JSON with [risk_rating, risk_types, risk_justification, reward_rating, reward_justification]
#
# Usage:
#   export OPENAI_API_KEY="..."
#   python occ_level_eval_all_computer_occ.py

from __future__ import annotations

import json
from typing import List, Dict, Any
from openai import OpenAI

MODEL = "gpt-5.4"  # change if needed (e.g., "gpt-5.1")

SYSTEM_MSG = (
    "You are a careful evaluator of expert–LLM collaboration. "
)

QUESTION = """\
For each occupation, determine the risks and rewards of using an LLM system (e.g., LLMs, LLM-based agents) in collaboration with a human expert for completing occupational tasks. For determination, use your knowledge of the current capabilities of the best-performing LLM systems. First, determine whether the occupation is safety-critical, i.e., any unintentional errors or exploitation of LLM capabilities by malicious actors can cause significant harm at the individual, organizational, or societal level. If yes, then quantify the extent to which using the combination of LLM Systems + Human Experts in performing the occupational tasks increases the risk compared to only having human experts complete occupational tasks. Considering the cost of detecting and mitigating risks with experts, determine if the combination of LLM systems + Human Experts can significantly decrease the cost (in USD) or time of completing the occupational tasks while keeping the same quality as expert humans.

For each evaluation:
- Rate both the reward and risk associated with LLM systems on a scale from 0 to 5. Here, for rewards, a rating of 0 means that using LLM systems cannot reduce overall completion time for occupational tasks or cost while keeping equivalent quality, rating <=2 means relatively low reduction in overall cost or time or both, rating of 3 means moderate reduction, and >=4 means high reduction. A rating of 0 for risks means that the occupation is not safety-critical, a risk rating <=2 means it can be easily detected and/or fixed by human experts, a rating of 3 means that the risk is moderate, and rating >=4 means the risk is high and cannot be easily detected.
- Provide a detailed justification in the format: [risks_rating, "risks_types", "risks_justification", rewards_rating, "rewards_justification"].
"""

OUTPUT_CONTRACT = """\
Return JSON only (no markdown, no extra text). Use this schema:
{
  "results": [
    {
      "occupation": "<string>",
      "is_safety_critical": <true/false>,
      "output": [<risk_rating number>, "<risks_types string>", "<risks_justification string>", <reward_rating number>, "<rewards_justification string>"]
    }
  ]
}
Notes:
- risks_types must be a concise comma-separated string (e.g., "Security, Privacy, Misinformation").
- If you cannot responsibly estimate USD savings, say so and provide qualitative/time-based assessment instead.
"""

OCCUPATIONS: List[str] = [
    "Computer Systems Analysts",
    "Health Informatics Specialists",
    "Information Security Analysts",
    "Computer and Information Research Scientists",
    "Computer Network Support Specialists",
    "Computer User Support Specialists",
    "Computer Network Architects",
    "Telecommunications Engineering Specialists",
    "Database Administrators",
    "Database Architects",
    "Data Warehousing Specialists",
    "Network and Computer Systems Administrators",
    "Computer Programmers",
    "Software Developers",
    "Software Quality Assurance Analysts and Testers",
    "Web Developers",
    "Web and Digital Interface Designers",
    "Video Game Designers",
    "Web Administrators",
    "Geographic Information Systems Technologists and Technicians",
    "Document Management Specialists",
    "Penetration Testers",
    "Information Security Engineers",
    "Digital Forensics Analysts",
    "Blockchain Engineers",
    "Computer Systems Engineers/Architects",
    "Information Technology Project Managers",
]

def build_user_prompt(occupations: List[str]) -> str:
    occ_block = "\n".join([f"- {o}" for o in occupations])
    return f"{QUESTION}\n\nOccupations:\n{occ_block}\n\n{OUTPUT_CONTRACT}"

def call_gpt5_occupation_level(occupations: List[str], reasoning_effort: str = "medium") -> Dict[str, Any]:
    client = OpenAI()
    prompt = build_user_prompt(occupations)

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": reasoning_effort},
    )

    text = resp.output_text.strip()

    # Best-effort JSON parsing in case the model adds minor stray whitespace/characters
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise

def postprocess(result: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a derived ranking/list of occupations where risk > reward, and where reward > risk."""
    risk_gt_reward = []
    reward_gt_risk = []
    unclear = []

    for r in result.get("results", []):
        occ = r.get("occupation", "")
        out = r.get("output", [])
        if not isinstance(out, list) or len(out) < 4:
            unclear.append(occ)
            continue
        risk = out[0]
        reward = out[3]
        if isinstance(risk, (int, float)) and isinstance(reward, (int, float)):
            if risk > reward:
                risk_gt_reward.append((occ, float(risk - reward)))
            elif reward > risk:
                reward_gt_risk.append((occ, float(reward - risk)))
            else:
                unclear.append(occ)
        else:
            unclear.append(occ)

    risk_gt_reward.sort(key=lambda x: x[1], reverse=True)
    reward_gt_risk.sort(key=lambda x: x[1], reverse=True)

    result["derived"] = {
        "risk>reward_ranked": [o for o, _ in risk_gt_reward],
        "reward>risk_ranked": [o for o, _ in reward_gt_risk],
        "unclear_or_unparsed": unclear,
        "risk_minus_reward_margins": {o: m for o, m in risk_gt_reward},
        "reward_minus_risk_margins": {o: m for o, m in reward_gt_risk},
    }
    return result

def main() -> None:
    result = call_gpt5_occupation_level(OCCUPATIONS, reasoning_effort="medium")
    result = postprocess(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()