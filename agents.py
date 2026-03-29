"""
NutriCoach ReAct — Reasoning + Acting Nutrition Agent System
=============================================================

Architecture: ReAct Pattern with LangGraph
-------------------------------------------

Each agent follows the ReAct loop:
    THINK → ACT (call tool) → OBSERVE (read result) → THINK → ...

The system has 5 specialist agents, each with their own tools, connected
through a LangGraph StateGraph with conditional routing and a Critic-Revise
feedback loop.

Graph:
                    ┌──────────────────┐
                    │   BMI AGENT      │  Tools: calculate_bmi, classify_risk
                    │  (ReAct loop)    │  Reasons about health profile
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  NUTRITION AGENT  │  Tools: compute_tdee, build_macro_targets
                    │  (ReAct loop)    │  conditional: standard vs intervention
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐ ◄──────────────┐
                    │  CRITIC AGENT    │  Tool: score_plan  LOOP (max 2)
                    │  (ReAct loop)    │  Scores 0-10, decides revise/proceed
                    └────────┬─────────┘ ────────────────┘
                             │ score ≥ 7
                    ┌────────▼─────────┐
                    │  MEAL AGENT      │  Tools: build_meal_plan, get_supplement_stack
                    │  (ReAct loop)    │  Builds 7-day plan + supplements
                    └────────▼─────────┘
                    ┌────────▼─────────┐
                    │  COACH AGENT     │  Tools: safety_check, synthesise_report
                    │  (ReAct loop)    │  Audits safety, writes final coaching report
                    └────────┬─────────┘
                             │
                            END

ReAct Pattern:
  Each agent has a system prompt that instructs it to reason step-by-step,
  call tools explicitly, read observations, and iterate until confident.
  Every agent's reasoning trace is captured and streamed to the UI.
"""

import os
import json
import math
from typing import TypedDict, Annotated, Sequence, Literal, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ═══════════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════════

class NutritionState(TypedDict):
    # User inputs
    age: int
    gender: str
    weight_kg: float
    height_cm: float
    activity_level: str
    goal: str
    dietary_restrictions: str
    health_conditions: str

    # Computed
    bmi: float
    bmi_category: str
    risk_level: str
    tdee: int

    # Loop control
    nutrition_revision_count: int
    nutrition_score: int

    # Message history (full ReAct traces per agent)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Agent ReAct traces (list of step dicts for UI rendering)
    bmi_trace: list
    nutrition_trace: list
    critic_trace: list
    meal_trace: list
    coach_trace: list

    # Final outputs
    bmi_report: str
    nutrition_plan: str
    nutrition_critique: str
    meal_plan: str
    supplement_advice: str
    final_report: str


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_model(api_key: str) -> ChatOpenAI:
    os.environ["OPENAI_API_KEY"] = api_key
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def _bmi(weight_kg: float, height_cm: float) -> float:
    h = height_cm / 100
    return round(weight_kg / (h * h), 1)

def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25.0: return "Normal weight"
    elif bmi < 30.0: return "Overweight"
    else:            return "Obese"

def _tdee(weight_kg, height_cm, age, gender, activity_level) -> int:
    if gender.lower() == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    mults = {"Sedentary": 1.2, "Lightly Active": 1.375,
             "Moderately Active": 1.55, "Very Active": 1.725, "Extremely Active": 1.9}
    return int(bmr * mults.get(activity_level, 1.55))

FMT = ("Write in plain text only. No markdown, no asterisks, no hashes. "
       "Section headings in ALL CAPS. All numbers must be specific to this patient.")

def _profile(s: NutritionState) -> str:
    return (f"Age {s['age']} | {s['gender']} | {s['weight_kg']}kg | {s['height_cm']}cm | "
            f"BMI {s['bmi']} ({s['bmi_category']}) | TDEE {s['tdee']} kcal | "
            f"Goal: {s['goal']} | Activity: {s['activity_level']} | "
            f"Diet: {s['dietary_restrictions']} | Conditions: {s['health_conditions']}")

def _run_react(model_with_tools, system_prompt: str, user_message: str,
               tools_by_name: dict) -> tuple[str, list]:
    """
    Run a ReAct loop: Think → Act → Observe → repeat until no more tool calls.
    Returns (final_text_response, trace_steps).
    trace_steps is a list of dicts:
        {"type": "thought"|"action"|"observation", "content": str, "tool": str|None}
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    trace = []
    max_iters = 6

    for _ in range(max_iters):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            # Extract thought from text content before tool call
            thought_text = response.content.strip() if response.content else "Determining which tool to use..."
            if thought_text:
                trace.append({"type": "thought", "content": thought_text, "tool": None})

            # Execute each tool call
            for tc in response.tool_calls:
                action_text = f"Calling {tc['name']} with: {json.dumps(tc['args'], indent=2)}"
                trace.append({"type": "action", "content": action_text, "tool": tc["name"]})

                if tc["name"] in tools_by_name:
                    try:
                        result = tools_by_name[tc["name"]].invoke(tc["args"])
                        obs_text = str(result)
                    except Exception as e:
                        obs_text = f"Tool error: {str(e)}"
                else:
                    obs_text = f"Tool '{tc['name']}' not found."

                trace.append({"type": "observation", "content": obs_text, "tool": tc["name"]})
                messages.append(ToolMessage(
                    content=obs_text,
                    name=tc["name"],
                    tool_call_id=tc["id"]
                ))
        else:
            # No tool calls — final answer
            final_text = response.content.strip()
            trace.append({"type": "thought", "content": "I have all the information needed. Composing final response.", "tool": None})
            return final_text, trace

    # Max iterations reached
    final = messages[-1].content if hasattr(messages[-1], 'content') else "Analysis complete."
    return final, trace


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — BMI AGENT
# Tools: calculate_bmi, classify_risk
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculate BMI and derive healthy weight range from height."""
    bmi = _bmi(weight_kg, height_cm)
    category = _bmi_category(bmi)
    h = height_cm / 100
    target_lo = round(18.5 * h * h, 1)
    target_hi = round(24.9 * h * h, 1)
    delta = round(abs(weight_kg - (target_hi if bmi > 25 else target_lo)), 1)
    weeks = max(4, int(delta / 0.5))
    return {
        "bmi": bmi,
        "category": category,
        "healthy_range_kg": f"{target_lo}–{target_hi}",
        "weight_change_needed_kg": delta,
        "estimated_weeks_at_safe_rate": weeks,
    }

@tool
def classify_risk(bmi: float, age: int, health_conditions: str) -> dict:
    """Classify cardiovascular and metabolic risk based on BMI, age, and conditions."""
    high_risk_conditions = ["diabetes", "hypertension", "heart", "kidney", "liver"]
    has_condition = any(c in health_conditions.lower() for c in high_risk_conditions)
    intervention = bmi < 17.5 or bmi >= 27.5 or age > 60 or has_condition

    cv_risk = "High" if (bmi >= 30 or has_condition) else "Moderate" if bmi >= 25 else "Low"
    metabolic_risk = "Elevated" if (bmi >= 27.5 or has_condition) else "Normal"

    return {
        "protocol": "intervention" if intervention else "standard",
        "cardiovascular_risk": cv_risk,
        "metabolic_risk": metabolic_risk,
        "intervention_triggered_by": (
            "BMI out of range" if (bmi < 17.5 or bmi >= 27.5) else
            "Age > 60" if age > 60 else
            f"Health condition: {health_conditions}" if has_condition else
            "None — standard protocol"
        ),
    }

BMI_TOOLS = [calculate_bmi, classify_risk]
BMI_TOOLS_BY_NAME = {t.name: t for t in BMI_TOOLS}

def bmi_agent_node(state: NutritionState, model: ChatOpenAI) -> dict:
    model_with_tools = model.bind_tools(BMI_TOOLS)

    system = f"""You are a clinical physiologist using the ReAct (Reasoning + Acting) framework.
You have access to tools: calculate_bmi, classify_risk.

{FMT}

REACT INSTRUCTIONS:
- First THINK about what you need to know
- Then ACT by calling the appropriate tool
- OBSERVE the result and reason about it
- Repeat until you have a complete clinical picture
- Finally write your assessment

Your goal: Produce a formal BMI and risk assessment for this patient."""

    user = f"""Patient profile:
Age: {state['age']} | Gender: {state['gender']}
Weight: {state['weight_kg']} kg | Height: {state['height_cm']} cm
Goal: {state['goal']} | Activity: {state['activity_level']}
Health conditions: {state['health_conditions']}

Use your tools to calculate BMI, classify risk, then write a formal clinical assessment with headings:
CLINICAL STATUS, TARGET ANALYSIS, RISK PROFILE, CLINICAL PRIORITIES."""

    final_text, trace = _run_react(model_with_tools, system, user, BMI_TOOLS_BY_NAME)

    # Extract computed values from tool observations
    bmi = _bmi(state["weight_kg"], state["height_cm"])
    cat = _bmi_category(bmi)
    tdee = _tdee(state["weight_kg"], state["height_cm"],
                 state["age"], state["gender"], state["activity_level"])

    risk_result = classify_risk.invoke({
        "bmi": bmi, "age": state["age"],
        "health_conditions": state["health_conditions"]
    })

    return {
        "bmi": bmi, "bmi_category": cat, "tdee": tdee,
        "risk_level": risk_result["protocol"],
        "bmi_report": final_text,
        "bmi_trace": trace,
        "nutrition_revision_count": 0,
        "nutrition_score": 0,
        "messages": [AIMessage(content=f"[BMI Agent] BMI {bmi} ({cat}). Protocol: {risk_result['protocol'].upper()}.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — NUTRITION AGENT
# Tools: compute_tdee, build_macro_targets
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def compute_tdee(weight_kg: float, height_cm: float, age: int,
                 gender: str, activity_level: str, goal: str) -> dict:
    """Compute TDEE using Mifflin-St Jeor and apply goal-based calorie adjustment."""
    tdee = _tdee(weight_kg, height_cm, age, gender, activity_level)
    adjustments = {"lose": -500, "maintain": 0, "gain": 350}
    adj = adjustments.get(goal.lower(), 0)
    min_floor = 1200 if gender.lower() == "female" else 1500
    target = max(min_floor, tdee + adj)
    return {
        "tdee_kcal": tdee,
        "goal_adjustment_kcal": adj,
        "daily_target_kcal": target,
        "calorie_floor_applied": target == min_floor,
        "deficit_or_surplus": abs(adj),
    }

@tool
def build_macro_targets(weight_kg: float, daily_kcal: int,
                        goal: str, risk_level: str) -> dict:
    """Calculate protein, carbohydrate, and fat targets based on goal and risk level."""
    if risk_level == "intervention":
        protein_g_per_kg = 2.0
    elif goal == "gain":
        protein_g_per_kg = 1.8
    else:
        protein_g_per_kg = 1.6

    protein_g = round(weight_kg * protein_g_per_kg)
    protein_kcal = protein_g * 4
    remaining = daily_kcal - protein_kcal
    fat_g = round((remaining * 0.30) / 9)
    carb_g = round((remaining * 0.70) / 4)

    return {
        "protein_g": protein_g,
        "protein_g_per_kg": protein_g_per_kg,
        "carbohydrate_g": carb_g,
        "fat_g": fat_g,
        "fibre_g_minimum": 25 if daily_kcal < 2000 else 30,
        "water_litres": round(weight_kg * 0.033, 1),
    }

NUTRITION_TOOLS = [compute_tdee, build_macro_targets]
NUTRITION_TOOLS_BY_NAME = {t.name: t for t in NUTRITION_TOOLS}

def nutrition_agent_node(state: NutritionState, model: ChatOpenAI) -> dict:
    model_with_tools = model.bind_tools(NUTRITION_TOOLS)
    is_intervention = state["risk_level"] == "intervention"

    revision_context = ""
    if state["nutrition_revision_count"] > 0:
        revision_context = f"\n\nPREVIOUS CRITIQUE TO ADDRESS:\n{state['nutrition_critique']}\nRevise your plan to fix every issue."

    system = f"""You are a Registered Dietitian using the ReAct framework.
You have tools: compute_tdee, build_macro_targets.
Protocol: {'INTERVENTION (therapeutic — 750 kcal max deficit/surplus, elevated protein)' if is_intervention else 'STANDARD (maintenance optimisation)'}

{FMT}
{revision_context}

REACT INSTRUCTIONS:
- THINK about what calculations you need
- ACT by calling tools in the right order (compute_tdee first, then build_macro_targets)
- OBSERVE each result carefully
- Build your complete nutrition prescription from the tool outputs"""

    user = f"""Patient: {_profile(state)}
BMI assessment: {state['bmi_report'][:400]}

Use your tools, then write a detailed nutrition plan with these ALL CAPS sections:
CALORIE PRESCRIPTION, MACRONUTRIENT TARGETS, MICRONUTRIENT PRIORITIES (4 nutrients),
HYDRATION, FOODS TO PRIORITISE (6 items), FOODS TO LIMIT (4 items), MEAL TIMING."""

    final_text, trace = _run_react(model_with_tools, system, user, NUTRITION_TOOLS_BY_NAME)

    return {
        "nutrition_plan": final_text,
        "nutrition_trace": trace,
        "messages": [AIMessage(content=f"[Nutrition Agent] Plan complete. Revision {state['nutrition_revision_count'] + 1}.")],
    }

def route_nutrition(state: NutritionState) -> Literal["nutrition_critic", "nutrition_critic"]:
    return "nutrition_critic"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — CRITIC AGENT (ReAct Feedback Loop)
# Tool: score_plan
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def score_plan(
    calorie_accuracy: int,
    macro_balance: int,
    clinical_safety: int,
    specificity: int,
    completeness: int,
) -> dict:
    """Score the nutrition plan on 5 dimensions (1-10 each) and compute overall."""
    scores = [calorie_accuracy, macro_balance, clinical_safety, specificity, completeness]
    overall = round(sum(scores) / len(scores))
    passed = overall >= 7
    weakest = min(zip(scores, ["calorie_accuracy","macro_balance","clinical_safety","specificity","completeness"]))
    return {
        "dimension_scores": {
            "calorie_accuracy": calorie_accuracy,
            "macro_balance": macro_balance,
            "clinical_safety": clinical_safety,
            "specificity": specificity,
            "completeness": completeness,
        },
        "overall_score": overall,
        "passed": passed,
        "weakest_dimension": weakest[1],
        "verdict": "PROCEED to meal planning" if passed else "REVISE — score below threshold of 7",
    }

CRITIC_TOOLS = [score_plan]
CRITIC_TOOLS_BY_NAME = {t.name: t for t in CRITIC_TOOLS}

def critic_agent_node(state: NutritionState, model: ChatOpenAI) -> dict:
    model_with_tools = model.bind_tools(CRITIC_TOOLS)

    system = f"""You are a senior clinical nutrition reviewer using the ReAct framework.
You have one tool: score_plan.

{FMT}

REACT INSTRUCTIONS:
- THINK: read the nutrition plan carefully and evaluate each dimension
- ACT: call score_plan with your scores (1-10 each)
- OBSERVE: read the verdict
- Write your detailed critique explaining every score and listing specific revision instructions"""

    user = f"""Patient: {_profile(state)}

NUTRITION PLAN TO REVIEW:
{state['nutrition_plan']}

Score these 5 dimensions honestly:
1. calorie_accuracy (1-10): Is TDEE calculated correctly and goal adjustment appropriate?
2. macro_balance (1-10): Are protein/carbs/fat targets correct for this patient?
3. clinical_safety (1-10): Is it safe for conditions: {state['health_conditions']}?
4. specificity (1-10): Are all numbers patient-specific (not generic ranges)?
5. completeness (1-10): Are all sections present and detailed?

After scoring, write your full critique with:
DIMENSION SCORES, OVERALL VERDICT, ISSUES FOUND, REVISION INSTRUCTIONS (or NONE if passed)."""

    final_text, trace = _run_react(model_with_tools, system, user, CRITIC_TOOLS_BY_NAME)

    # Parse score from tool observation in trace
    score = 7  # safe default
    for step in trace:
        if step["type"] == "observation" and "overall_score" in step["content"]:
            try:
                obs = json.loads(step["content"].replace("'", '"'))
                score = obs.get("overall_score", 7)
            except Exception:
                import re
                m = re.search(r"overall_score['\": ]+(\d+)", step["content"])
                if m:
                    score = int(m.group(1))

    count = state["nutrition_revision_count"] + 1
    return {
        "nutrition_critique": final_text,
        "nutrition_score": score,
        "nutrition_revision_count": count,
        "critic_trace": trace,
        "messages": [AIMessage(content=f"[Critic Agent] Score: {score}/10. Revision count: {count}.")],
    }

def should_revise(state: NutritionState) -> Literal["nutrition_agent", "meal_agent"]:
    if state["nutrition_score"] < 7 and state["nutrition_revision_count"] < 2:
        return "nutrition_agent"
    return "meal_agent"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — MEAL AGENT
# Tools: build_meal_plan, get_supplement_stack
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def build_meal_plan(
    daily_kcal: int,
    protein_g: int,
    carb_g: int,
    fat_g: int,
    dietary_restrictions: str,
    goal: str,
) -> dict:
    """Generate a 7-day meal plan framework with per-meal calorie distribution."""
    meal_split = {"breakfast": 0.25, "lunch": 0.30, "dinner": 0.30, "snack": 0.15}
    cuisines = ["Mediterranean", "Asian Fusion", "Middle Eastern", "Latin American",
                "Indian", "Japanese", "West African"]
    days = []
    for i, cuisine in enumerate(cuisines, 1):
        days.append({
            "day": i,
            "cuisine_theme": cuisine,
            "meal_targets": {
                meal: {
                    "kcal": round(daily_kcal * pct),
                    "protein_g": round(protein_g * pct),
                    "carb_g": round(carb_g * pct),
                    "fat_g": round(fat_g * pct),
                }
                for meal, pct in meal_split.items()
            },
            "daily_total_kcal": daily_kcal,
        })
    return {
        "plan_framework": days,
        "dietary_restrictions_applied": dietary_restrictions,
        "goal": goal,
        "note": "Use this framework to write specific meal names and ingredients.",
    }

@tool
def get_supplement_stack(
    bmi: float,
    age: int,
    gender: str,
    goal: str,
    health_conditions: str,
    dietary_restrictions: str,
) -> dict:
    """Generate an evidence-based supplement stack for this patient profile."""
    stack = []

    # Core supplements everyone gets
    stack.append({
        "name": "Vitamin D3",
        "dose": "2000 IU/day",
        "timing": "With breakfast (fat-soluble)",
        "evidence": "Strong",
        "rationale": "Widespread deficiency; supports immunity and bone health"
    })
    stack.append({
        "name": "Omega-3 Fish Oil",
        "dose": "1000–2000 mg EPA+DHA/day",
        "timing": "With a meal",
        "evidence": "Strong",
        "rationale": "Reduces inflammation; cardiovascular and cognitive support"
    })

    # Goal-specific
    if goal == "lose":
        stack.append({
            "name": "Magnesium Glycinate",
            "dose": "300–400 mg/day",
            "timing": "Before bed",
            "evidence": "Moderate",
            "rationale": "Supports sleep quality and insulin sensitivity during caloric deficit"
        })
    elif goal == "gain":
        stack.append({
            "name": "Creatine Monohydrate",
            "dose": "5 g/day",
            "timing": "Post-workout or anytime",
            "evidence": "Strong",
            "rationale": "Most evidence-backed supplement for muscle strength and mass"
        })

    # Condition-specific
    if "diabetes" in health_conditions.lower():
        stack.append({
            "name": "Berberine",
            "dose": "500 mg with meals (2–3x/day)",
            "timing": "With meals",
            "evidence": "Moderate",
            "rationale": "Clinically shown to improve insulin sensitivity"
        })
    if "hypertension" in health_conditions.lower():
        stack.append({
            "name": "CoQ10",
            "dose": "100–200 mg/day",
            "timing": "With a meal",
            "evidence": "Moderate",
            "rationale": "May modestly reduce blood pressure; antioxidant support"
        })

    # Vegan-specific
    if "vegan" in dietary_restrictions.lower():
        stack.append({
            "name": "Vitamin B12",
            "dose": "1000 mcg/day (methylcobalamin)",
            "timing": "Morning",
            "evidence": "Strong",
            "rationale": "Critical — not available in plant foods"
        })

    return {"supplements": stack[:5], "total_supplements": min(len(stack), 5)}

MEAL_TOOLS = [build_meal_plan, get_supplement_stack]
MEAL_TOOLS_BY_NAME = {t.name: t for t in MEAL_TOOLS}

def meal_agent_node(state: NutritionState, model: ChatOpenAI) -> dict:
    model_with_tools = model.bind_tools(MEAL_TOOLS)

    # Extract calorie/macro from nutrition plan (use computed values as fallback)
    daily_kcal = state["tdee"] + (-500 if state["goal"] == "lose" else 350 if state["goal"] == "gain" else 0)
    min_floor = 1200 if state["gender"].lower() == "female" else 1500
    daily_kcal = max(min_floor, daily_kcal)
    protein_g = round(state["weight_kg"] * (2.0 if state["risk_level"] == "intervention" else 1.6))
    carb_g = round((daily_kcal - protein_g * 4) * 0.70 / 4)
    fat_g = round((daily_kcal - protein_g * 4) * 0.30 / 9)

    system = f"""You are a culinary nutritionist and supplementation specialist using the ReAct framework.
You have tools: build_meal_plan, get_supplement_stack.

{FMT}

REACT INSTRUCTIONS:
- THINK about what you need to build
- ACT by calling build_meal_plan first (to get the framework), then get_supplement_stack
- OBSERVE each result — use the framework numbers to write specific meals
- Write out all 7 days with actual dish names, ingredients, and macros"""

    user = f"""Patient: {_profile(state)}
Nutrition prescription: {state['nutrition_plan'][:500]}

Computed targets: {daily_kcal} kcal/day | Protein {protein_g}g | Carbs {carb_g}g | Fat {fat_g}g

Call build_meal_plan with these targets, then get_supplement_stack.

After observing the results, write:

SECTION 1: 7-DAY MEAL PLAN
For each day use:
DAY N — [Cuisine Theme]
BREAKFAST: [dish] | Ingredients: [list] | Macros: P Xg / C Xg / F Xg / ~X kcal
LUNCH: [dish] | Ingredients: [list] | Macros: P Xg / C Xg / F Xg / ~X kcal
DINNER: [dish] | Ingredients: [list] | Macros: P Xg / C Xg / F Xg / ~X kcal
SNACK: [item] | Macros: P Xg / C Xg / F Xg / ~X kcal
DAILY TOTAL: X kcal

SECTION 2: SUPPLEMENT PROTOCOL
List each supplement from the tool result with dose, timing, and rationale."""

    final_text, trace = _run_react(model_with_tools, system, user, MEAL_TOOLS_BY_NAME)

    # Split sections
    meal_part = final_text
    supp_part = ""
    if "SECTION 2:" in final_text:
        parts = final_text.split("SECTION 2:", 1)
        meal_part = parts[0].replace("SECTION 1:", "").strip()
        supp_part = parts[1].strip()

    return {
        "meal_plan": meal_part,
        "supplement_advice": supp_part,
        "meal_trace": trace,
        "messages": [AIMessage(content="[Meal Agent] 7-day plan + supplements complete.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — COACH AGENT
# Tools: safety_check, synthesise_report
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def safety_check(
    daily_kcal: int,
    protein_g: int,
    weight_kg: float,
    gender: str,
    health_conditions: str,
) -> dict:
    """Audit the nutrition plan for clinical safety violations."""
    violations = []
    min_floor = 1200 if gender.lower() == "female" else 1500
    max_protein = round(2.5 * weight_kg)

    if daily_kcal < min_floor:
        violations.append(f"Calories {daily_kcal} kcal below safe floor of {min_floor} kcal — increase immediately.")
    if protein_g > max_protein:
        violations.append(f"Protein {protein_g}g exceeds safe ceiling of {max_protein}g for {weight_kg}kg patient.")
    if "kidney" in health_conditions.lower() and protein_g > round(1.2 * weight_kg):
        violations.append(f"High protein contraindicated for kidney disease. Reduce to max {round(1.2 * weight_kg)}g.")

    return {
        "status": "CLEARED" if not violations else "FLAGS RAISED",
        "violations": violations,
        "violations_count": len(violations),
        "safe_calorie_floor": min_floor,
        "safe_protein_ceiling": max_protein,
    }

@tool
def synthesise_report(
    nutrition_score: int,
    nutrition_revisions: int,
    safety_status: str,
    risk_level: str,
    goal: str,
) -> dict:
    """Generate a quality and process summary for the coaching report."""
    quality_label = "Excellent" if nutrition_score >= 9 else "Good" if nutrition_score >= 7 else "Acceptable"
    return {
        "quality_summary": f"Nutrition plan scored {nutrition_score}/10 ({quality_label}) after {nutrition_revisions} revision(s).",
        "safety_summary": f"Safety audit: {safety_status}.",
        "protocol": risk_level.upper(),
        "goal": goal,
        "process_note": (
            f"This plan was generated by 5 specialist ReAct agents, each reasoning "
            f"step-by-step with tools before producing their outputs."
        ),
    }

COACH_TOOLS = [safety_check, synthesise_report]
COACH_TOOLS_BY_NAME = {t.name: t for t in COACH_TOOLS}

def coach_agent_node(state: NutritionState, model: ChatOpenAI) -> dict:
    model_with_tools = model.bind_tools(COACH_TOOLS)

    # Estimate daily kcal for safety check
    daily_kcal = state["tdee"] + (-500 if state["goal"] == "lose" else 350 if state["goal"] == "gain" else 0)
    min_floor = 1200 if state["gender"].lower() == "female" else 1500
    daily_kcal = max(min_floor, daily_kcal)
    protein_g = round(state["weight_kg"] * (2.0 if state["risk_level"] == "intervention" else 1.6))

    system = f"""You are a senior wellness coach and clinical safety officer using the ReAct framework.
You have tools: safety_check, synthesise_report.

{FMT}

REACT INSTRUCTIONS:
- THINK about what to verify before writing the final report
- ACT: call safety_check first, then synthesise_report
- OBSERVE results and incorporate safety status into your coaching
- Write the final integrated coaching report"""

    user = f"""Patient: {_profile(state)}
Nutrition plan score: {state['nutrition_score']}/10 after {state['nutrition_revision_count']} revision(s)

Run safety_check with: daily_kcal={daily_kcal}, protein_g={protein_g},
weight_kg={state['weight_kg']}, gender={state['gender']}, health_conditions={state['health_conditions']}

Then run synthesise_report with the process metadata.

After observing both results, write the final coaching report:

EXECUTIVE SUMMARY (3 sentences: current status, what this plan achieves, timeline)
SAFETY AUDIT RESULT (what was checked and the outcome)
YOUR THREE PRIORITIES THIS WEEK (specific and actionable)
WHAT TO TRACK WEEKLY (4 metrics with how to measure each)
YOUR 90-DAY MILESTONES (Week 2, Month 1, Month 3)
COACHING NOTE (warm, professional closing — end with: Your specialist team has designed this plan with precision for you. The work starts now.)"""

    final_text, trace = _run_react(model_with_tools, system, user, COACH_TOOLS_BY_NAME)

    return {
        "final_report": final_text,
        "coach_trace": trace,
        "messages": [AIMessage(content="[Coach Agent] Final report complete.")],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(api_key: str):
    model = _make_model(api_key)

    g = StateGraph(NutritionState)

    g.add_node("bmi_agent",       lambda s: bmi_agent_node(s, model))
    g.add_node("nutrition_agent", lambda s: nutrition_agent_node(s, model))
    g.add_node("nutrition_critic",lambda s: critic_agent_node(s, model))
    g.add_node("meal_agent",      lambda s: meal_agent_node(s, model))
    g.add_node("coach_agent",     lambda s: coach_agent_node(s, model))

    g.set_entry_point("bmi_agent")
    g.add_edge("bmi_agent",        "nutrition_agent")
    g.add_edge("nutrition_agent",  "nutrition_critic")

    # Critic feedback loop
    g.add_conditional_edges(
        "nutrition_critic",
        should_revise,
        {"nutrition_agent": "nutrition_agent", "meal_agent": "meal_agent"},
    )

    g.add_edge("meal_agent",  "coach_agent")
    g.add_edge("coach_agent", END)

    return g.compile()


def make_initial_state(**kw) -> NutritionState:
    return NutritionState(
        age=kw["age"], gender=kw["gender"],
        weight_kg=kw["weight_kg"], height_cm=kw["height_cm"],
        activity_level=kw["activity_level"], goal=kw["goal"],
        dietary_restrictions=kw.get("dietary_restrictions", "none"),
        health_conditions=kw.get("health_conditions", "none"),
        bmi=0.0, bmi_category="", risk_level="standard", tdee=0,
        nutrition_revision_count=0, nutrition_score=0,
        messages=[],
        bmi_trace=[], nutrition_trace=[], critic_trace=[],
        meal_trace=[], coach_trace=[],
        bmi_report="", nutrition_plan="", nutrition_critique="",
        meal_plan="", supplement_advice="", final_report="",
    )
