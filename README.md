# NutriCoach AI v2

A multi-agent clinical nutrition system built with LangGraph. Eleven specialist agents collaborate through a non-linear graph with three feedback loops, conditional routing, parallel execution, and a dedicated safety resolution layer.



<img width="2560" height="1600" alt="Screenshot (407)" src="https://github.com/user-attachments/assets/bb284f55-1522-4450-a395-9585592ec0f3" />

---

## Architecture

```
SUPERVISOR
    │  dispatches all agents, sets protocol, computes TDEE
    ▼
BMI ANALYST ──► RISK CLASSIFIER
                     │
          ┌──────────┴──────────┐
          │                     │
    STANDARD               INTERVENTION        ← conditional route
    NUTRITION              NUTRITION             (based on BMI + age + conditions)
    PLANNER                PLANNER
          │                     │
          └──────────┬──────────┘
                     │
              NUTRITION CRITIC ◄──────────────┐
                     │                        │
              score ≥ 7?  NO ─────────────────┘  Loop 1 (max 2 revisions)
                     │ YES
                     ▼
         ┌───────────────────────┐
         │      PARALLEL         │
         │  Meal Planner         │  ← concurrent threads
         │  Supplement Advisor   │
         └───────────┬───────────┘
                     │
               MEAL CRITIC ◄────────────────────┐
                     │                          │
              score ≥ 7?  NO ──────────────────┘  Loop 2 (max 2 revisions)
                     │ YES
                     ▼
               RISK GUARD
                     │
           flags raised?  YES ──► SAFETY RESOLVER  Loop 3 (conditional)
                     │                    │
                     └──────────┬─────────┘
                                │
                        WELLNESS COACH
                                │
                               END
```

---

## The Three Feedback Loops

### Loop 1 — Nutrition Critique

After the Nutrition Planner produces a plan, the Nutrition Critic scores it across five dimensions: calorie accuracy, macronutrient balance, clinical appropriateness, specificity, and completeness. If the overall score is below 7/10 and fewer than two revisions have occurred, the Planner is called again with the critique included as context. The loop runs until the plan passes or the maximum iteration count is reached.

### Loop 2 — Meal Critique

The Meal Critic evaluates the 7-day meal plan against the same nutrition targets. It checks calorie alignment per day, macro compliance, dietary restriction adherence, variety, and prep practicality. Below 7/10 triggers a full revision of both the meal plan and supplement protocol (since they run in parallel). Maximum two iterations.

### Loop 3 — Safety Resolution

The Risk Guard audits the full protocol for clinical safety violations: calorie floors (1,200 kcal for women, 1,500 kcal for men), protein ceilings (2.5 g/kg), supplement upper limits, and condition-specific contraindications. If any flags are raised, the Safety Resolver produces a corrected plan that patches each violation before the Wellness Coach synthesises the final report.

---

## Agents

| # | Agent | Role |
|---|-------|------|
| 1 | **Supervisor** | Validates inputs, computes BMR/TDEE, assigns protocol, logs dispatch plan |
| 2 | **BMI Analyst** | Clinical body composition assessment with timeline projection |
| 3 | **Risk Classifier** | Stratifies cardiovascular and metabolic risk, routes the graph |
| 4a | **Standard Nutrition Planner** | Full Mifflin-St Jeor TDEE prescription for BMI 17.5–27.4 |
| 4b | **Intervention Nutrition Planner** | Therapeutic protocol with 750 kcal deficit cap and meal timing |
| 5 | **Nutrition Critic** | Scores plan 0–10 across 5 dimensions, triggers Loop 1 |
| 6 | **Meal Planner** | 7-day meal plan with per-meal macros (runs parallel) |
| 7 | **Supplement Advisor** | 6-supplement evidence-based protocol (runs parallel) |
| 8 | **Meal Critic** | Scores meal plan 0–10 across 5 dimensions, triggers Loop 2 |
| 9 | **Risk Guard** | Clinical safety audit — checks floors, ceilings, contraindications |
| 10 | **Safety Resolver** | Patches violations raised by Risk Guard (Loop 3, conditional) |
| 11 | **Wellness Coach** | Final synthesis — executive summary, action plan, 90-day milestones |

---

## Routing Logic

```python
risk = "intervention" if (
    bmi < 17.5 or bmi >= 27.5 or age > 60 or
    any health condition in ["diabetes", "hypertension", "heart", "kidney", "liver"]
) else "standard"
```

The Supervisor computes this before any agent runs. The Risk Classifier confirms it with clinical reasoning.

---

## State Design

All eleven agents share one `NutritionState` TypedDict. Loop counters prevent infinite loops. The Supervisor pre-populates computed fields (BMI, TDEE, risk level) so every downstream agent has accurate data from the start.

```python
class NutritionState(TypedDict):
    # User inputs
    age, gender, weight_kg, height_cm
    activity_level, goal
    dietary_restrictions, health_conditions

    # Computed by Supervisor
    bmi, bmi_category, risk_level, tdee

    # Loop control
    nutrition_revision_count   # Loop 1 counter
    meal_revision_count        # Loop 2 counter
    nutrition_score            # Loop 1 gate (threshold: 7)
    meal_score                 # Loop 2 gate (threshold: 7)
    safety_resolved            # Loop 3 flag

    # Agent outputs (11 fields)
    supervisor_log, bmi_report, risk_assessment
    nutrition_plan, nutrition_critique
    meal_plan, meal_critique
    supplement_advice
    risk_flags, safety_resolution
    final_report
```

---

## Project Structure

```
nutricoach_v2/
├── agents.py         All 11 agent definitions + LangGraph state machine
├── app.py            Flask server with SSE streaming
├── requirements.txt
└── templates/
    └── index.html    Bootstrap 5 dark interface with live graph
```

---

## Setup

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

Enter your OpenAI API key, fill in the profile, click **Generate Clinical Plan**. Watch the execution graph animate as each agent activates. The execution log tracks every agent start, loop decision, and routing choice in real time.

---

## Key LangGraph Patterns Used

**Conditional edges** route the graph based on state values at runtime.

**Feedback loops** are implemented with counter fields in state and conditional edge functions that check both the score and the iteration count before deciding whether to re-enter or exit the loop.

**Parallel execution** wraps two agent calls in a ThreadPoolExecutor and merges their outputs into a single state update before the next node.

**Safety gate** uses a conditional edge that only activates the Safety Resolver node when the Risk Guard raises flags — otherwise routing directly to the Wellness Coach.

---

## Disclaimer

For educational and portfolio purposes only. Not a substitute for advice from a registered dietitian or physician.

---

## License

MIT
