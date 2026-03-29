"""
NutriCoach AI v2 — Flask Application
"""

import json
from flask import Flask, render_template, request, Response, stream_with_context, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    d = request.json
    api_key = d.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "OpenAI API key required"}), 400

    params = dict(
        age=int(d.get("age", 28)),
        gender=d.get("gender", "Male"),
        weight_kg=float(d.get("weight_kg", 78)),
        height_cm=float(d.get("height_cm", 172)),
        activity_level=d.get("activity_level", "Moderately Active"),
        goal=d.get("goal", "lose"),
        dietary_restrictions=d.get("dietary_restrictions", "none"),
        health_conditions=d.get("health_conditions", "none"),
    )

    def generate():
        try:
            from agents import (
                _make_model, _bmi, _bmi_category, _tdee, make_initial_state,
                supervisor_node, bmi_analyst_node, risk_classifier_node,
                standard_nutrition_node, intervention_nutrition_node,
                nutrition_critic_node, should_revise_nutrition,
                parallel_meal_and_supplement, meal_critic_node, should_revise_meal,
                risk_guard_node, should_resolve_safety,
                safety_resolver_node, coach_node,
            )

            def emit(**kw):
                return f"data: {json.dumps(kw)}\n\n"

            model = _make_model(api_key)
            state = make_initial_state(**params)

            # ── Quick preview ──
            bmi = _bmi(params["weight_kg"], params["height_cm"])
            cat = _bmi_category(bmi)
            tdee = _tdee(params["weight_kg"], params["height_cm"],
                         params["age"], params["gender"], params["activity_level"])
            yield emit(event="preview", bmi=bmi, category=cat, tdee=tdee)

            # ── Supervisor ──
            yield emit(event="agent_start", agent="Supervisor", node="supervisor")
            out = supervisor_node(state, model)
            state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
            yield emit(event="agent_done", agent="Supervisor", node="supervisor",
                       content=state["supervisor_log"],
                       risk_level=state["risk_level"], tdee=state["tdee"])

            # ── BMI Analyst ──
            yield emit(event="agent_start", agent="BMI Analyst", node="bmi_analyst")
            out = bmi_analyst_node(state, model)
            state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
            yield emit(event="agent_done", agent="BMI Analyst", node="bmi_analyst",
                       content=state["bmi_report"])

            # ── Risk Classifier ──
            yield emit(event="agent_start", agent="Risk Classifier", node="risk_classifier")
            out = risk_classifier_node(state, model)
            state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
            yield emit(event="agent_done", agent="Risk Classifier", node="risk_classifier",
                       content=state["risk_assessment"], risk_level=state["risk_level"])

            # ── Nutrition loop ──
            while True:
                variant = "standard" if state["risk_level"] == "standard" else "intervention"
                agent_name = "Standard Nutrition Planner" if variant == "standard" else "Intervention Nutrition Planner"
                yield emit(event="agent_start", agent=agent_name, node="nutrition_planner",
                           variant=variant, revision=state["nutrition_revision_count"])

                if variant == "standard":
                    out = standard_nutrition_node(state, model)
                else:
                    out = intervention_nutrition_node(state, model)
                state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
                yield emit(event="agent_done", agent=agent_name, node="nutrition_planner",
                           content=state["nutrition_plan"], variant=variant)

                # Nutrition Critic
                yield emit(event="agent_start", agent="Nutrition Critic", node="nutrition_critic",
                           revision=state["nutrition_revision_count"])
                out = nutrition_critic_node(state, model)
                state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
                yield emit(event="agent_done", agent="Nutrition Critic", node="nutrition_critic",
                           content=state["nutrition_critique"],
                           score=state["nutrition_score"],
                           revision=state["nutrition_revision_count"])

                decision = should_revise_nutrition(state)
                yield emit(event="loop_decision", loop="nutrition",
                           decision=decision,
                           score=state["nutrition_score"],
                           revision=state["nutrition_revision_count"])
                if decision == "proceed_to_meals":
                    break

            # ── Meal + Supplement loop ──
            while True:
                yield emit(event="agent_start", agent="Meal Planner", node="meal_planner",
                           revision=state["meal_revision_count"])
                yield emit(event="agent_start", agent="Supplement Advisor", node="supplement_advisor",
                           revision=state["meal_revision_count"])
                out = parallel_meal_and_supplement(state, model)
                state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
                yield emit(event="agent_done", agent="Meal Planner", node="meal_planner",
                           content=state["meal_plan"])
                yield emit(event="agent_done", agent="Supplement Advisor", node="supplement_advisor",
                           content=state["supplement_advice"])

                # Meal Critic
                yield emit(event="agent_start", agent="Meal Critic", node="meal_critic",
                           revision=state["meal_revision_count"])
                out = meal_critic_node(state, model)
                state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
                yield emit(event="agent_done", agent="Meal Critic", node="meal_critic",
                           content=state["meal_critique"],
                           score=state["meal_score"],
                           revision=state["meal_revision_count"])

                decision = should_revise_meal(state)
                yield emit(event="loop_decision", loop="meal",
                           decision=decision,
                           score=state["meal_score"],
                           revision=state["meal_revision_count"])
                if decision == "proceed_to_safety":
                    break

            # ── Risk Guard ──
            yield emit(event="agent_start", agent="Risk Guard", node="risk_guard")
            out = risk_guard_node(state, model)
            state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
            cleared = state["risk_flags"].startswith("CLEARANCE GRANTED")
            yield emit(event="agent_done", agent="Risk Guard", node="risk_guard",
                       content=state["risk_flags"], cleared=cleared)

            # ── Safety Resolver (conditional) ──
            if not cleared:
                yield emit(event="agent_start", agent="Safety Resolver", node="safety_resolver")
                out = safety_resolver_node(state, model)
                state.update(out); state["messages"] = list(state["messages"]) + list(out.get("messages",[]))
                yield emit(event="agent_done", agent="Safety Resolver", node="safety_resolver",
                           content=state["safety_resolution"])

            # ── Wellness Coach ──
            yield emit(event="agent_start", agent="Wellness Coach", node="wellness_coach")
            out = coach_node(state, model)
            state.update(out)
            yield emit(event="agent_done", agent="Wellness Coach", node="wellness_coach",
                       content=state["final_report"])

            yield emit(event="complete",
                       nutrition_revisions=state["nutrition_revision_count"] - 1,
                       meal_revisions=state["meal_revision_count"] - 1,
                       safety_resolved=state["safety_resolved"])

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'trace': traceback.format_exc()})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
