"""
NutriCoach ReAct — Flask Application
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
                _bmi, _bmi_category, _tdee, make_initial_state, build_graph,
                bmi_agent_node, nutrition_agent_node, critic_agent_node,
                meal_agent_node, coach_agent_node, should_revise,
                _make_model,
            )

            def emit(**kw):
                return f"data: {json.dumps(kw)}\n\n"

            # Quick preview
            bmi = _bmi(params["weight_kg"], params["height_cm"])
            cat = _bmi_category(bmi)
            tdee = _tdee(params["weight_kg"], params["height_cm"],
                         params["age"], params["gender"], params["activity_level"])
            yield emit(event="preview", bmi=bmi, category=cat, tdee=tdee)

            model = _make_model(api_key)
            state = make_initial_state(**params)

            # ── Agent 1: BMI Agent ──────────────────────────────────────
            yield emit(event="agent_start", agent="BMI Agent",
                       description="Calculates BMI, classifies risk, assigns protocol")
            out = bmi_agent_node(state, model)
            state.update(out)
            state["messages"] = list(state["messages"]) + list(out.get("messages", []))
            yield emit(event="agent_done", agent="BMI Agent",
                       content=state["bmi_report"],
                       trace=state["bmi_trace"],
                       bmi=state["bmi"], category=state["bmi_category"],
                       risk_level=state["risk_level"], tdee=state["tdee"])

            # ── Nutrition loop ──────────────────────────────────────────
            while True:
                yield emit(event="agent_start", agent="Nutrition Agent",
                           description=f"{'Intervention' if state['risk_level']=='intervention' else 'Standard'} nutrition prescription",
                           revision=state["nutrition_revision_count"])
                out = nutrition_agent_node(state, model)
                state.update(out)
                state["messages"] = list(state["messages"]) + list(out.get("messages", []))
                yield emit(event="agent_done", agent="Nutrition Agent",
                           content=state["nutrition_plan"],
                           trace=state["nutrition_trace"],
                           variant=state["risk_level"])

                yield emit(event="agent_start", agent="Critic Agent",
                           description="Scores plan 0-10 using score_plan tool · triggers revision if score < 7",
                           revision=state["nutrition_revision_count"])
                out = critic_agent_node(state, model)
                state.update(out)
                state["messages"] = list(state["messages"]) + list(out.get("messages", []))
                yield emit(event="agent_done", agent="Critic Agent",
                           content=state["nutrition_critique"],
                           trace=state["critic_trace"],
                           score=state["nutrition_score"],
                           revision=state["nutrition_revision_count"])

                decision = should_revise(state)
                yield emit(event="loop_decision",
                           decision=decision,
                           score=state["nutrition_score"],
                           revision=state["nutrition_revision_count"])

                if decision == "meal_agent":
                    break

            # ── Agent 4: Meal Agent ──────────────────────────────────────
            yield emit(event="agent_start", agent="Meal Agent",
                       description="Builds 7-day meal plan + supplement stack using tools")
            out = meal_agent_node(state, model)
            state.update(out)
            state["messages"] = list(state["messages"]) + list(out.get("messages", []))
            yield emit(event="agent_done", agent="Meal Agent",
                       meal_content=state["meal_plan"],
                       supp_content=state["supplement_advice"],
                       trace=state["meal_trace"])

            # ── Agent 5: Coach Agent ─────────────────────────────────────
            yield emit(event="agent_start", agent="Coach Agent",
                       description="Runs safety_check + synthesise_report tools · writes final coaching report")
            out = coach_agent_node(state, model)
            state.update(out)
            yield emit(event="agent_done", agent="Coach Agent",
                       content=state["final_report"],
                       trace=state["coach_trace"])

            yield emit(event="complete",
                       nutrition_revisions=state["nutrition_revision_count"] - 1,
                       nutrition_score=state["nutrition_score"])

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'event':'error','message':str(e),'trace':traceback.format_exc()})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
