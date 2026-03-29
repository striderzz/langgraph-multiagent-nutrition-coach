"""
NutriCoach AI v2 — Multi-Agent System with Feedback Loops
==========================================================

Full Architecture:
                                      ┌──────────────┐
                            ┌─────────│  SUPERVISOR  │────────────┐
                            │         └──────┬───────┘            │
                            │                │ dispatch            │
                            │    ┌───────────▼────────────┐       │
                            │    │    BMI + RISK LAYER     │       │
                            │    │  ┌──────┐  ┌────────┐  │       │
                            │    │  │ BMI  │→ │ Risk   │  │       │
                            │    │  │Agent │  │Classif.│  │       │
                            │    │  └──────┘  └────┬───┘  │       │
                            │    └────────────────┬┘       │       │
                            │                     │         │       │
                            │      ┌──────────────▼──────────────┐ │
                            │      │     NUTRITION LAYER          │ │
                            │      │  ┌─────────┐ ┌───────────┐  │ │
                            │      │  │Standard │ │Intervention│  │ │
                            │      │  │Planner  │ │ Planner   │  │ │
                            │      │  └────┬────┘ └─────┬─────┘  │ │
                            │      │       └──────┬──────┘        │ │
                            │      │              │ nutrition_plan │ │
                            │      │    ┌─────────▼──────────┐    │ │
                            │      │    │  Nutrition Critic   │◄───┼─┘  LOOP 1
                            │      │    │  (review + score)   │    │    max 2 iters
                            │      │    └─────────┬──────────┘    │
                            │      │    score≥7?  │ NO → revise   │
                            │      └──────────────┘               │
                            │                                      │
                            │      ┌───────────────────────────────┐
                            │      │       MEAL + SUPP LAYER        │
                            │      │  ┌──────────┐ ┌───────────┐   │
                            │      │  │  Meal    │ │Supplement │   │ Parallel
                            │      │  │ Planner  │ │ Advisor   │   │
                            │      │  └────┬─────┘ └────┬──────┘   │
                            │      │       └──────┬──────┘          │
                            │      │    ┌─────────▼──────────┐      │
                            │      │    │   Meal Critic       │◄─────┘  LOOP 2
                            │      │    │  (quality check)    │         max 2 iters
                            │      │    └─────────┬──────────┘
                            │      │    score≥7?  │ NO → revise
                            │      └──────────────┘
                            │
                            │      ┌──────────────────────────┐
                            │      │      SAFETY LAYER         │
                            │      │  ┌───────────────────┐    │
                            │      │  │   Risk Guard       │    │
                            │      │  │  (safety audit)    │    │
                            │      │  └─────────┬──────────┘    │
                            │      │  ┌─────────▼──────────┐    │
                            │      │  │  Safety Resolver    │    │  LOOP 3
                            │      │  │ (fixes violations)  │    │  if flags raised
                            │      │  └─────────────────────┘    │
                            │      └──────────────────────────────┘
                            │
                            └──────────────────────────────────────────┐
                                                                        ▼
                                                             ┌──────────────────┐
                                                             │  Wellness Coach   │
                                                             │  (synthesiser)    │
                                                             └──────────┬────────┘
                                                                        ▼
                                                                      END

Loops:
  Loop 1 — Nutrition Critique:  Nutrition Critic scores the plan (0-10).
            If score < 7, Planner revises. Max 2 iterations.
  Loop 2 — Meal Critique:       Meal Critic scores the 7-day plan (0-10).
            If score < 7, Meal Planner revises. Max 2 iterations.
  Loop 3 — Safety Resolution:   If Risk Guard raises flags, Safety Resolver
            patches the plan and clears violations. Always 1 resolution pass.
"""

import os
import json
import concurrent.futures
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

class NutritionState(TypedDict):
    # ── User inputs ──────────────────────────────────────────────────────────
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    activity_level: str
    goal: str
    dietary_restrictions: str
    health_conditions: str          # e.g. "type 2 diabetes, hypertension"

    # ── Computed ─────────────────────────────────────────────────────────────
    bmi: float
    bmi_category: str
    risk_level: str                 # "standard" | "intervention"
    tdee: int                       # calculated TDEE in kcal

    # ── Loop counters ─────────────────────────────────────────────────────────
    nutrition_revision_count: int   # Loop 1: how many times plan was revised
    meal_revision_count: int        # Loop 2: how many times meal plan was revised
    nutrition_score: int            # Loop 1: critic score 0-10
    meal_score: int                 # Loop 2: critic score 0-10
    safety_resolved: bool           # Loop 3: whether safety resolver ran

    # ── Message history ───────────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Agent outputs ─────────────────────────────────────────────────────────
    bmi_report: str
    risk_assessment: str
    nutrition_plan: str
    nutrition_critique: str
    meal_plan: str
    meal_critique: str
    supplement_advice: str
    risk_flags: str
    safety_resolution: str
    final_report: str

    # ── Supervisor tracking ───────────────────────────────────────────────────
    supervisor_log: str             # log of supervisor decisions


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_model(api_key: str, temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=temperature)

def _bmi(weight_kg: float, height_cm: float) -> float:
    h = height_cm / 100
    return round(weight_kg / (h * h), 1)

def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25.0: return "Normal weight"
    elif bmi < 30.0: return "Overweight"
    else:            return "Obese"

def _tdee(weight_kg, height_cm, age, gender, activity_level) -> int:
    """Mifflin-St Jeor + activity multiplier."""
    if gender.lower() == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    mults = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extremely Active": 1.9,
    }
    m = mults.get(activity_level, 1.55)
    return int(bmr * m)

FMT = (
    "FORMATTING — mandatory: plain text only, no asterisks, no hash symbols, "
    "no markdown. Section headings in ALL CAPS. Lists use plain dashes or numbers. "
    "All figures must be patient-specific, not ranges."
)

def _profile(s: NutritionState) -> str:
    return (
        f"Age {s['age']} | {s['gender']} | {s['weight_kg']} kg | "
        f"{s['height_cm']} cm | BMI {s['bmi']} ({s['bmi_category']}) | "
        f"TDEE {s['tdee']} kcal | Goal: {s['goal']} | "
        f"Activity: {s['activity_level']} | "
        f"Dietary restrictions: {s['dietary_restrictions']} | "
        f"Health conditions: {s['health_conditions']}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_node(state: NutritionState, model: ChatOpenAI) -> dict:
    """
    Supervisor orchestrates execution and maintains a decision log.
    Runs at the start to validate inputs and set strategy.
    """
    bmi = _bmi(state["weight_kg"], state["height_cm"])
    cat = _bmi_category(bmi)
    tdee = _tdee(
        state["weight_kg"], state["height_cm"],
        state["age"], state["gender"], state["activity_level"]
    )

    risk = "intervention" if (
        bmi < 17.5 or bmi >= 27.5 or
        state["age"] > 60 or
        any(c in state["health_conditions"].lower()
            for c in ["diabetes","hypertension","heart","kidney","liver","eating disorder"])
    ) else "standard"

    log = (
        f"SUPERVISOR DISPATCH LOG\n"
        f"Patient: {_profile({**state, 'bmi': bmi, 'bmi_category': cat, 'tdee': tdee})}\n"
        f"Protocol assigned: {risk.upper()}\n"
        f"TDEE calculated: {tdee} kcal/day\n"
        f"Nutrition critique loop: enabled (max 2 iterations)\n"
        f"Meal critique loop: enabled (max 2 iterations)\n"
        f"Safety resolution loop: enabled (triggered if flags raised)\n"
        f"Agents dispatched: BMI Analyst → Risk Classifier → "
        f"{'Standard' if risk == 'standard' else 'Intervention'} Nutrition Planner → "
        f"Nutrition Critic [loop] → Meal Planner + Supplement Advisor (parallel) → "
        f"Meal Critic [loop] → Risk Guard → Safety Resolver [conditional] → Wellness Coach"
    )

    return {
        "bmi": bmi,
        "bmi_category": cat,
        "tdee": tdee,
        "risk_level": risk,
        "supervisor_log": log,
        "nutrition_revision_count": 0,
        "meal_revision_count": 0,
        "nutrition_score": 0,
        "meal_score": 0,
        "safety_resolved": False,
        "messages": [AIMessage(content=f"[Supervisor] Protocol: {risk.upper()}. TDEE: {tdee} kcal. Dispatching agents.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — BMI ANALYST
# ═══════════════════════════════════════════════════════════════════════════════

def bmi_analyst_node(state: NutritionState, model: ChatOpenAI) -> dict:
    h = state["height_cm"] / 100
    target_lo = round(18.5 * h * h, 1)
    target_hi = round(24.9 * h * h, 1)
    delta = round(abs(state["weight_kg"] - (target_hi if state["bmi"] > 25 else target_lo)), 1)
    weeks = max(4, int(delta / 0.5))

    prompt = f"""{FMT}

You are a clinical physiologist. Write a formal body composition assessment.

PATIENT: {_profile(state)}
Healthy weight range: {target_lo}–{target_hi} kg | Weight delta: {delta} kg | Est. weeks: {weeks}

Write with these ALL CAPS headings:

CLINICAL STATUS:
BODY COMPOSITION ANALYSIS:
TARGET ANALYSIS:
TIMELINE PROJECTION:
CLINICAL PRIORITIES:"""

    r = model.invoke([
        SystemMessage(content="You are a board-certified clinical physiologist."),
        HumanMessage(content=prompt),
    ])
    return {
        "bmi_report": r.content.strip(),
        "messages": [AIMessage(content=f"[BMI Analyst] BMI {state['bmi']} — {state['bmi_category']}. Complete.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — RISK CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def risk_classifier_node(state: NutritionState, model: ChatOpenAI) -> dict:
    prompt = f"""{FMT}

You are a clinical risk stratification specialist.

PATIENT: {_profile(state)}
BMI report: {state['bmi_report'][:400]}

Write a risk assessment with ALL CAPS headings:

CARDIOVASCULAR RISK:
METABOLIC RISK FACTORS:
CONDITION-SPECIFIC CONSIDERATIONS:
PROTOCOL JUSTIFICATION:
RECOMMENDED SCREENING:"""

    r = model.invoke([
        SystemMessage(content="You are a clinical risk stratification specialist."),
        HumanMessage(content=prompt),
    ])
    return {
        "risk_assessment": r.content.strip(),
        "messages": [AIMessage(content=f"[Risk Classifier] Protocol: {state['risk_level'].upper()}. Complete.")],
    }

def route_by_risk(state: NutritionState) -> Literal["standard_nutrition", "intervention_nutrition"]:
    return "standard_nutrition" if state["risk_level"] == "standard" else "intervention_nutrition"


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — NUTRITION PLANNERS
# ═══════════════════════════════════════════════════════════════════════════════

def standard_nutrition_node(state: NutritionState, model: ChatOpenAI) -> dict:
    revision_note = ""
    if state["nutrition_revision_count"] > 0:
        revision_note = f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{state['nutrition_critique']}\nRevise accordingly."

    prompt = f"""{FMT}

You are a Registered Dietitian.{revision_note}

PATIENT: {_profile(state)}
BMI report context: {state['bmi_report'][:300]}

Produce a precision nutrition plan with these ALL CAPS headings:

BMR CALCULATION: (Mifflin-St Jeor — show arithmetic)
TDEE: (activity multiplier — show arithmetic, result: {state['tdee']} kcal)
CALORIE TARGET: (adjusted for goal — show delta, justify magnitude)
PROTEIN TARGET: (g/day and g/kg — show calculation)
CARBOHYDRATE TARGET: (g/day, timing strategy)
DIETARY FAT TARGET: (g/day minimum floor)
FIBRE TARGET: (g/day)
HYDRATION: (litres/day)
MICRONUTRIENT PRIORITIES: (4 nutrients, dose, source)
FOODS TO PRIORITISE: (8 foods with one-line rationale)
FOODS TO LIMIT: (5 items with explanation)
MEAL FREQUENCY: (recommended eating pattern)"""

    r = model.invoke([
        SystemMessage(content="You are a Registered Dietitian writing clinical nutrition prescriptions."),
        HumanMessage(content=prompt),
    ])
    return {
        "nutrition_plan": r.content.strip(),
        "messages": [AIMessage(content=f"[Standard Nutrition Planner] Revision {state['nutrition_revision_count'] + 1}. Complete.")],
    }


def intervention_nutrition_node(state: NutritionState, model: ChatOpenAI) -> dict:
    revision_note = ""
    if state["nutrition_revision_count"] > 0:
        revision_note = f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{state['nutrition_critique']}\nRevise accordingly."

    prompt = f"""{FMT}

You are a clinical dietitian specialising in therapeutic nutrition.{revision_note}

PATIENT: {_profile(state)}
Risk assessment: {state['risk_assessment'][:350]}

Produce a therapeutic nutrition plan with these ALL CAPS headings:

BMR CALCULATION: (Mifflin-St Jeor — show arithmetic)
TDEE: (activity multiplier — result: {state['tdee']} kcal)
THERAPEUTIC CALORIE TARGET: (max 750 kcal deficit / 500 kcal surplus — justify choice)
PROTEIN PRESCRIPTION: (elevated for lean mass — show g/kg × weight = g/day)
CARBOHYDRATE PROTOCOL: (glycaemic management, timing)
FAT PRESCRIPTION: (minimum hormonal floor)
FIBRE TARGET: (satiety and gut health)
MEAL TIMING PROTOCOL: (eating window, pre/post workout)
METABOLIC SUPPORT MICRONUTRIENTS: (5 nutrients with clinical doses)
HYDRATION AND ELECTROLYTES:
CLINICAL RED FLAGS: (5 specific behaviours to avoid)
PROGRESS BENCHMARKS: (week 2, month 1, month 3)
MEDICAL REFERRAL TRIGGERS: (symptoms requiring physician escalation)"""

    r = model.invoke([
        SystemMessage(content="You are a clinical dietitian specialising in therapeutic weight management."),
        HumanMessage(content=prompt),
    ])
    return {
        "nutrition_plan": r.content.strip(),
        "messages": [AIMessage(content=f"[Intervention Nutrition Planner] Revision {state['nutrition_revision_count'] + 1}. Complete.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP 1 — NUTRITION CRITIC
# ═══════════════════════════════════════════════════════════════════════════════

def nutrition_critic_node(state: NutritionState, model: ChatOpenAI) -> dict:
    prompt = f"""{FMT}

You are a senior clinical nutrition reviewer. Critically evaluate the nutrition plan below.

PATIENT: {_profile(state)}

PLAN TO REVIEW:
{state['nutrition_plan']}

Score the plan on these dimensions (1-10 each):
1. Calorie accuracy (correct TDEE calculation and appropriate adjustment)
2. Macronutrient balance (correct ratios for goal and risk level)
3. Clinical appropriateness (safe for this patient's conditions and age)
4. Specificity (all numbers patient-specific, no vague ranges)
5. Completeness (all required sections present and detailed)

Format your response:

DIMENSION SCORES:
- Calorie accuracy: X/10 — reason
- Macronutrient balance: X/10 — reason
- Clinical appropriateness: X/10 — reason
- Specificity: X/10 — reason
- Completeness: X/10 — reason

OVERALL SCORE: X/10

CRITICAL ISSUES: (list any score below 7 with specific correction needed)

REVISION INSTRUCTIONS: (concrete changes the planner must make — or write NONE if score >= 7)"""

    r = model.invoke([
        SystemMessage(content="You are a senior clinical nutrition reviewer providing rigorous critique."),
        HumanMessage(content=prompt),
    ])

    content = r.content.strip()
    # Parse overall score
    score = 7  # default pass
    for line in content.splitlines():
        if "OVERALL SCORE:" in line:
            try:
                score = int(line.split(":")[1].strip().split("/")[0].strip())
            except Exception:
                pass

    count = state["nutrition_revision_count"] + 1
    return {
        "nutrition_critique": content,
        "nutrition_score": score,
        "nutrition_revision_count": count,
        "messages": [AIMessage(content=f"[Nutrition Critic] Score: {score}/10. Revision count: {count}.")],
    }

def should_revise_nutrition(state: NutritionState) -> Literal["revise_nutrition", "proceed_to_meals"]:
    """Loop 1: revise if score < 7 and under 2 iterations."""
    if state["nutrition_score"] < 7 and state["nutrition_revision_count"] < 2:
        return "revise_nutrition"
    return "proceed_to_meals"


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — MEAL PLANNER + SUPPLEMENT ADVISOR (parallel)
# ═══════════════════════════════════════════════════════════════════════════════

def meal_plan_node(state: NutritionState, model: ChatOpenAI) -> dict:
    revision_note = ""
    if state["meal_revision_count"] > 0:
        revision_note = f"\n\nMEAL CRITIQUE TO ADDRESS:\n{state['meal_critique']}\nRevise accordingly."

    prompt = f"""{FMT}

You are a culinary nutritionist.{revision_note}

PATIENT: {_profile(state)}
Nutrition prescription: {state['nutrition_plan'][:700]}

Design a 7-day meal plan. Use this exact structure for every day:

DAY N — [Cuisine Theme]

BREAKFAST: [dish]
  Ingredients: [quantities]
  Macros: Protein Xg | Carbs Xg | Fat Xg | ~X kcal
  Prep: X min

LUNCH: [dish]
  Ingredients: [quantities]
  Macros: Protein Xg | Carbs Xg | Fat Xg | ~X kcal
  Prep: X min

DINNER: [dish]
  Ingredients: [quantities]
  Macros: Protein Xg | Carbs Xg | Fat Xg | ~X kcal
  Prep: X min

SNACK: [item] — Protein Xg | Carbs Xg | Fat Xg | ~X kcal

DAILY TOTAL: X kcal | Protein Xg | Carbs Xg | Fat Xg

Seven different cuisine themes. Respect dietary restrictions exactly."""

    r = model.invoke([
        SystemMessage(content="You are a culinary nutritionist and meal planning specialist."),
        HumanMessage(content=prompt),
    ])
    return {
        "meal_plan": r.content.strip(),
        "messages": [AIMessage(content=f"[Meal Planner] Revision {state['meal_revision_count'] + 1}. Complete.")],
    }


def supplement_node(state: NutritionState, model: ChatOpenAI) -> dict:
    prompt = f"""{FMT}

You are a clinical supplementation specialist.

PATIENT: {_profile(state)}
Risk level: {state['risk_level']}

Design a 6-supplement evidence-based protocol. For each:

SUPPLEMENT N: [Name] — [Form]
  CLINICAL RATIONALE: specific to this patient
  EVIDENCE LEVEL: Strong/Moderate/Emerging with brief explanation
  DOSE: X mg/IU/g per day
  TIMING: when and why
  DURATION: X weeks then reassess / ongoing
  SAFETY NOTES: contraindications, upper limits, drug interactions
  COST: ~$X/month

End with one paragraph labelled MEDICAL DISCLAIMER:"""

    r = model.invoke([
        SystemMessage(content="You are a board-certified sports dietitian and supplementation specialist."),
        HumanMessage(content=prompt),
    ])
    return {
        "supplement_advice": r.content.strip(),
        "messages": [AIMessage(content="[Supplement Advisor] Protocol complete.")],
    }


def parallel_meal_and_supplement(state: NutritionState, model: ChatOpenAI) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_meal = ex.submit(meal_plan_node, state, model)
        f_supp = ex.submit(supplement_node, state, model)
        meal_out = f_meal.result()
        supp_out = f_supp.result()
    return {
        "meal_plan":         meal_out["meal_plan"],
        "supplement_advice": supp_out["supplement_advice"],
        "messages":          list(meal_out["messages"]) + list(supp_out["messages"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP 2 — MEAL CRITIC
# ═══════════════════════════════════════════════════════════════════════════════

def meal_critic_node(state: NutritionState, model: ChatOpenAI) -> dict:
    prompt = f"""{FMT}

You are a clinical dietitian and culinary expert reviewing a 7-day meal plan.

PATIENT: {_profile(state)}
NUTRITION TARGETS: {state['nutrition_plan'][:400]}

MEAL PLAN TO REVIEW:
{state['meal_plan'][:1500]}

Score on these dimensions (1-10 each):
1. Calorie alignment (daily totals match prescription within 100 kcal)
2. Macro compliance (protein/carbs/fat match targets)
3. Dietary restriction adherence (zero violations)
4. Variety and palatability (diverse, appealing, realistic)
5. Prep practicality (achievable for working professional)

Format:

DIMENSION SCORES:
- Calorie alignment: X/10 — reason
- Macro compliance: X/10 — reason
- Dietary restriction adherence: X/10 — reason
- Variety and palatability: X/10 — reason
- Prep practicality: X/10 — reason

OVERALL SCORE: X/10

VIOLATIONS FOUND: (specific meals with issues — or NONE)

REVISION INSTRUCTIONS: (exact changes needed — or NONE if score >= 7)"""

    r = model.invoke([
        SystemMessage(content="You are a clinical dietitian and culinary expert reviewing meal plans."),
        HumanMessage(content=prompt),
    ])

    content = r.content.strip()
    score = 7
    for line in content.splitlines():
        if "OVERALL SCORE:" in line:
            try:
                score = int(line.split(":")[1].strip().split("/")[0].strip())
            except Exception:
                pass

    count = state["meal_revision_count"] + 1
    return {
        "meal_critique": content,
        "meal_score": score,
        "meal_revision_count": count,
        "messages": [AIMessage(content=f"[Meal Critic] Score: {score}/10. Revision count: {count}.")],
    }

def should_revise_meal(state: NutritionState) -> Literal["revise_meal", "proceed_to_safety"]:
    """Loop 2: revise if score < 7 and under 2 iterations."""
    if state["meal_score"] < 7 and state["meal_revision_count"] < 2:
        return "revise_meal"
    return "proceed_to_safety"


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — RISK GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def risk_guard_node(state: NutritionState, model: ChatOpenAI) -> dict:
    min_kcal = 1200 if state["gender"].lower() == "female" else 1500
    max_protein = round(2.5 * state["weight_kg"])

    prompt = f"""{FMT}

You are a clinical safety officer auditing a nutrition protocol.

PATIENT: {_profile(state)}
Safety thresholds: Calorie floor {min_kcal} kcal | Protein ceiling {max_protein} g/day

NUTRITION PLAN:
{state['nutrition_plan'][:600]}

SUPPLEMENT PROTOCOL:
{state['supplement_advice'][:400]}

MEAL PLAN EXCERPT:
{state['meal_plan'][:400]}

Check for violations:
1. Calories below {min_kcal} kcal/day
2. Protein above {max_protein} g/day without justification
3. Supplement doses exceeding established upper limits
4. Recommendations contraindicated for: {state['health_conditions']}
5. Missing nutrients critical for age {state['age']} and BMI {state['bmi']}

If no violations: first line must be exactly: CLEARANCE GRANTED
If violations found: first line must be exactly: SAFETY FLAGS IDENTIFIED
Then list each violation with heading and specific correction."""

    r = model.invoke([
        SystemMessage(content="You are a clinical safety officer reviewing nutrition protocols."),
        HumanMessage(content=prompt),
    ])

    flags = r.content.strip()
    cleared = flags.startswith("CLEARANCE GRANTED")
    return {
        "risk_flags": flags,
        "messages": [AIMessage(content=f"[Risk Guard] {'Cleared.' if cleared else 'Flags raised.'}")],
    }

def should_resolve_safety(state: NutritionState) -> Literal["safety_resolver", "wellness_coach"]:
    """Loop 3: run resolver only if flags were raised."""
    if not state["risk_flags"].startswith("CLEARANCE GRANTED"):
        return "safety_resolver"
    return "wellness_coach"


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP 3 — SAFETY RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def safety_resolver_node(state: NutritionState, model: ChatOpenAI) -> dict:
    prompt = f"""{FMT}

You are a clinical dietitian tasked with resolving safety violations in a nutrition protocol.

PATIENT: {_profile(state)}

SAFETY FLAGS RAISED:
{state['risk_flags']}

ORIGINAL NUTRITION PLAN:
{state['nutrition_plan'][:700]}

Produce a corrected nutrition plan summary that:
1. Addresses every specific flag raised
2. States what was changed and why
3. Confirms the corrected values are now within safe limits

Use ALL CAPS headings:

VIOLATIONS RESOLVED:
CORRECTED CALORIE TARGET: (if applicable)
CORRECTED PROTEIN TARGET: (if applicable)
CORRECTED SUPPLEMENT NOTES: (if applicable)
CONDITION-SPECIFIC CORRECTIONS: (if applicable)
SAFETY CONFIRMATION: (confirm all values now within clinical limits)"""

    r = model.invoke([
        SystemMessage(content="You are a clinical dietitian resolving protocol safety issues."),
        HumanMessage(content=prompt),
    ])
    return {
        "safety_resolution": r.content.strip(),
        "safety_resolved": True,
        "messages": [AIMessage(content="[Safety Resolver] Violations patched. Resolution complete.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — WELLNESS COACH (final synthesiser)
# ═══════════════════════════════════════════════════════════════════════════════

def coach_node(state: NutritionState, model: ChatOpenAI) -> dict:
    safety_note = ""
    if state.get("safety_resolved"):
        safety_note = f"\n\nSAFETY CORRECTIONS APPLIED:\n{state['safety_resolution']}\nIncorporate these corrections.\n"

    quality_note = (
        f"Nutrition plan quality score: {state['nutrition_score']}/10 "
        f"(after {state['nutrition_revision_count']} revision(s)). "
        f"Meal plan quality score: {state['meal_score']}/10 "
        f"(after {state['meal_revision_count']} revision(s))."
    )

    prompt = f"""{FMT}

You are a senior certified health and wellness coach delivering a final integrated coaching report.

PATIENT: {_profile(state)}{safety_note}

QUALITY ASSURANCE: {quality_note}

You have reviewed all specialist reports. Write the final coaching report with ALL CAPS headings:

EXECUTIVE SUMMARY:
(3 sentences: where the patient stands today, what this plan will achieve, time horizon)

SYSTEM OVERVIEW:
(Brief note on the multi-agent process: {state['nutrition_revision_count']} nutrition revision(s),
{state['meal_revision_count']} meal revision(s), safety audit {'with corrections applied' if state.get('safety_resolved') else 'passed with no flags'})

YOUR THREE PRIORITIES THIS WEEK:
1. (specific, tied to nutrition plan)
2. (specific, tied to meal preparation)
3. (specific, tied to supplement or hydration protocol)

WHAT TO TRACK WEEKLY:
(4 metrics — what, how, what good looks like)

YOUR 90-DAY MILESTONES:
Week 2: (specific)
Month 1: (specific)
Month 3: (specific — approaching or reaching goal)

COACHING NOTE:
(One paragraph — warm, direct, professional. Acknowledge the journey ahead.
End with exactly: Your specialist team has designed this plan with precision for you. The work starts now.)"""

    r = model.invoke([
        SystemMessage(content="You are a senior certified health coach writing formal client reports."),
        HumanMessage(content=prompt),
    ])
    return {
        "final_report": r.content.strip(),
        "messages": [AIMessage(content="[Wellness Coach] Final report complete.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_nutrition_graph(api_key: str):
    model = _make_model(api_key)

    g = StateGraph(NutritionState)

    # Register all nodes
    g.add_node("supervisor",             lambda s: supervisor_node(s, model))
    g.add_node("bmi_analyst",            lambda s: bmi_analyst_node(s, model))
    g.add_node("risk_classifier",        lambda s: risk_classifier_node(s, model))
    g.add_node("standard_nutrition",     lambda s: standard_nutrition_node(s, model))
    g.add_node("intervention_nutrition", lambda s: intervention_nutrition_node(s, model))
    g.add_node("nutrition_critic",       lambda s: nutrition_critic_node(s, model))
    g.add_node("parallel_agents",        lambda s: parallel_meal_and_supplement(s, model))
    g.add_node("meal_critic",            lambda s: meal_critic_node(s, model))
    g.add_node("risk_guard",             lambda s: risk_guard_node(s, model))
    g.add_node("safety_resolver",        lambda s: safety_resolver_node(s, model))
    g.add_node("wellness_coach",         lambda s: coach_node(s, model))

    # Entry
    g.set_entry_point("supervisor")

    # Supervisor → BMI Analyst → Risk Classifier
    g.add_edge("supervisor",      "bmi_analyst")
    g.add_edge("bmi_analyst",     "risk_classifier")

    # Risk Classifier → conditional nutrition planner
    g.add_conditional_edges(
        "risk_classifier",
        route_by_risk,
        {
            "standard_nutrition":     "standard_nutrition",
            "intervention_nutrition": "intervention_nutrition",
        }
    )

    # Both planners → Nutrition Critic
    g.add_edge("standard_nutrition",     "nutrition_critic")
    g.add_edge("intervention_nutrition", "nutrition_critic")

    # LOOP 1: Nutrition Critic → revise OR proceed
    g.add_conditional_edges(
        "nutrition_critic",
        should_revise_nutrition,
        {
            "revise_nutrition":    "standard_nutrition" if True else "intervention_nutrition",
            "proceed_to_meals":    "parallel_agents",
        }
    )

    # Parallel agents → Meal Critic
    g.add_edge("parallel_agents", "meal_critic")

    # LOOP 2: Meal Critic → revise OR proceed
    g.add_conditional_edges(
        "meal_critic",
        should_revise_meal,
        {
            "revise_meal":        "parallel_agents",
            "proceed_to_safety":  "risk_guard",
        }
    )

    # Risk Guard → conditional safety resolver or coach
    g.add_conditional_edges(
        "risk_guard",
        should_resolve_safety,
        {
            "safety_resolver": "safety_resolver",
            "wellness_coach":  "wellness_coach",
        }
    )

    # LOOP 3: Safety Resolver → Coach
    g.add_edge("safety_resolver", "wellness_coach")
    g.add_edge("wellness_coach",  END)

    return g.compile()


def make_initial_state(**kw) -> NutritionState:
    return NutritionState(
        age=kw["age"], gender=kw["gender"],
        weight_kg=kw["weight_kg"], height_cm=kw["height_cm"],
        activity_level=kw["activity_level"], goal=kw["goal"],
        dietary_restrictions=kw.get("dietary_restrictions", "none"),
        health_conditions=kw.get("health_conditions", "none"),
        bmi=0.0, bmi_category="", risk_level="standard", tdee=0,
        nutrition_revision_count=0, meal_revision_count=0,
        nutrition_score=0, meal_score=0, safety_resolved=False,
        messages=[], supervisor_log="",
        bmi_report="", risk_assessment="",
        nutrition_plan="", nutrition_critique="",
        meal_plan="", meal_critique="",
        supplement_advice="", risk_flags="",
        safety_resolution="", final_report="",
    )
