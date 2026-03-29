"""
visualize_graph.py — NutriCoach ReAct Agent Graph
==================================================
Renders the 5-agent ReAct LangGraph as:
  1. Interactive Pyvis HTML  (nutricoach_react_graph.html)
  2. Static Matplotlib PNG   (nutricoach_react_graph.png)

Each node shows:
  - Agent name
  - Tools available
  - ReAct pattern (Think → Act → Observe)

Usage:
    python visualize_graph.py [--out-dir ./output] [--no-open]
"""

import os
import json
import webbrowser


# ─── Graph definition ────────────────────────────────────────────────────────

NODES = [
    {
        "id":    "bmi_agent",
        "label": "BMI Agent",
        "tools": "calculate_bmi\nclassify_risk",
        "color": "#2ecc71",
        "x":     80,   "y": 220,
        "desc":  "Calculates BMI from weight/height.\nClassifies cardiovascular risk.\nAssigns Standard or Intervention protocol.\nReAct: THINK → ACT → OBSERVE",
    },
    {
        "id":    "nutrition_agent",
        "label": "Nutrition Agent",
        "tools": "compute_tdee\nbuild_macro_targets",
        "color": "#27ae60",
        "x":     260,  "y": 220,
        "desc":  "Computes TDEE (Mifflin-St Jeor).\nBuilds macro targets from tool output.\nStandard vs Intervention protocol.\nReAct: THINK → ACT → OBSERVE",
    },
    {
        "id":    "nutrition_critic",
        "label": "Critic Agent",
        "tools": "score_plan",
        "color": "#3498db",
        "x":     440,  "y": 220,
        "desc":  "Scores plan 0-10 on 5 dimensions:\ncalorie accuracy, macro balance,\nclinical safety, specificity, completeness.\nTriggers revision if score < 7 (max 2 iters).\nReAct: THINK → ACT → OBSERVE",
    },
    {
        "id":    "meal_agent",
        "label": "Meal Agent",
        "tools": "build_meal_plan\nget_supplement_stack",
        "color": "#9b59b6",
        "x":     620,  "y": 220,
        "desc":  "Builds 7-day meal framework from tools.\nGenerates evidence-based supplement stack.\nWrites specific meals using tool outputs.\nReAct: THINK → ACT → OBSERVE",
    },
    {
        "id":    "coach_agent",
        "label": "Coach Agent",
        "tools": "safety_check\nsynthesize_report",
        "color": "#e67e22",
        "x":     800,  "y": 220,
        "desc":  "Audits plan for clinical safety violations.\nGenerates process quality summary.\nWrites final integrated coaching report.\nReAct: THINK → ACT → OBSERVE",
    },
]

EDGES = [
    ("bmi_agent",       "nutrition_agent", "bmi + risk profile",    False),
    ("nutrition_agent", "nutrition_critic","nutrition plan",         False),
    ("nutrition_critic","nutrition_agent", "score < 7 → revise",    True),   # loop back
    ("nutrition_critic","meal_agent",      "score ≥ 7 → proceed",   False),
    ("meal_agent",      "coach_agent",     "meal plan + supplements",False),
]

EDGE_COLORS = {
    ("bmi_agent",       "nutrition_agent"): "#2ecc71",
    ("nutrition_agent", "nutrition_critic"):"#27ae60",
    ("nutrition_critic","nutrition_agent"): "#3498db",  # loop
    ("nutrition_critic","meal_agent"):      "#27ae60",
    ("meal_agent",      "coach_agent"):     "#9b59b6",
}

TOOL_NODES = [
    {"id": "t_calc_bmi",    "label": "calculate_bmi()",    "parent": "bmi_agent",        "color": "#1a8a45", "x": 20,  "y": 340},
    {"id": "t_risk",        "label": "classify_risk()",    "parent": "bmi_agent",        "color": "#1a8a45", "x": 140, "y": 340},
    {"id": "t_tdee",        "label": "compute_tdee()",     "parent": "nutrition_agent",  "color": "#155d30", "x": 200, "y": 340},
    {"id": "t_macros",      "label": "build_macro_targets()","parent":"nutrition_agent", "color": "#155d30", "x": 320, "y": 340},
    {"id": "t_score",       "label": "score_plan()",       "parent": "nutrition_critic", "color": "#1a5276", "x": 440, "y": 340},
    {"id": "t_meal",        "label": "build_meal_plan()",  "parent": "meal_agent",       "color": "#5b2c6f", "x": 560, "y": 340},
    {"id": "t_supp",        "label": "get_supplement_stack()","parent":"meal_agent",     "color": "#5b2c6f", "x": 680, "y": 340},
    {"id": "t_safety",      "label": "safety_check()",     "parent": "coach_agent",      "color": "#784212", "x": 740, "y": 340},
    {"id": "t_synth",       "label": "synthesize_report()","parent": "coach_agent",      "color": "#784212", "x": 860, "y": 340},
]


# ─── Try to import the actual LangGraph compiled graph ──────────────────────

def _try_get_langgraph(out_dir: str):
    """Attempt to render LangGraph's native Mermaid PNG."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agents import build_graph
        # Build graph with a dummy key just to compile the structure
        # (no actual API calls — we only call .compile())
        g = build_graph.__wrapped__(None) if hasattr(build_graph, '__wrapped__') else None
        if g is None:
            return
        png_bytes = g.get_graph().draw_mermaid_png()
        path = os.path.join(out_dir, "nutricoach_mermaid.png")
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"[viz] LangGraph Mermaid PNG → {path}")
    except Exception as e:
        print(f"[viz] Mermaid PNG skipped ({e})")


# ─── Matplotlib PNG ──────────────────────────────────────────────────────────

def save_matplotlib(output_path: str = "nutricoach_react_graph.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError:
        print("[viz] matplotlib not installed — skipping PNG.")
        return

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor("#04100a")
    ax.set_facecolor("#04100a")
    ax.set_xlim(-40, 920)
    ax.set_ylim(-20, 480)
    ax.axis("off")

    ax.set_title(
        "NutriCoach ReAct — 5-Agent LangGraph System\n"
        "Think → Act (tool) → Observe loop inside each agent",
        fontsize=12, fontweight="bold",
        color="#b8e6c8", fontfamily="monospace", pad=16,
    )

    # ── Draw tool nodes ──────────────────────────────────────────────────
    for t in TOOL_NODES:
        bw, bh = 140, 28
        x, y = t["x"], 360
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.04",
            facecolor=t["color"] + "40",
            edgecolor=t["color"],
            lw=1.2, zorder=2,
        ))
        ax.text(x, y, t["label"], ha="center", va="center",
                fontsize=7, fontweight="600", color=t["color"],
                fontfamily="monospace", zorder=3)

    # ── Draw agent nodes ──────────────────────────────────────────────────
    BW, BH = 140, 64
    node_pos = {n["id"]: (n["x"], n["y"]) for n in NODES}

    for n in NODES:
        x, y = n["x"], n["y"]
        c = n["color"]
        ax.add_patch(FancyBboxPatch(
            (x - BW/2, y - BH/2), BW, BH,
            boxstyle="round,pad=0.05",
            facecolor=c + "18", edgecolor=c, lw=2, zorder=3,
        ))
        ax.text(x, y - 10, n["label"],
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=c,
                fontfamily="monospace", zorder=4)
        ax.text(x, y + 6, n["tools"],
                ha="center", va="center",
                fontsize=7, color=c + "bb",
                fontfamily="monospace", zorder=4,
                multialignment="center")
        # ReAct badge
        ax.text(x, y + 22,
                "T → A → O",
                ha="center", va="center",
                fontsize=6.5, color="#2ecc71" + "80",
                fontfamily="monospace", zorder=4)

        # Dashed line from agent to its tools
        for t in TOOL_NODES:
            if t["parent"] == n["id"]:
                ax.plot([x, t["x"]], [y - BH/2 - 2, 360 + 14],
                        color=c + "40", lw=1, linestyle="--", zorder=1)

    # ── Draw edges ────────────────────────────────────────────────────────
    for src, dst, lbl, is_loop in EDGES:
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[dst]
        ec = EDGE_COLORS.get((src, dst), "#2ecc71")

        if is_loop:
            # arc above for loop-back
            ax.annotate(
                "", xy=(x1 + BW/2 + 4, y1 - 8),
                xytext=(x0 - BW/2 - 4, y0 - 8),
                arrowprops=dict(
                    arrowstyle="-|>", color=ec, lw=1.4, mutation_scale=10,
                    connectionstyle="arc3,rad=-0.45",
                ),
            )
            ax.text((x0 + x1) / 2, y0 - 52,
                    lbl, ha="center", fontsize=7, color=ec,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#04100a", ec="none"))
        else:
            pad = BW / 2 + 6
            dx = x1 - x0
            length = abs(dx)
            ux = dx / length if length else 1
            xs, xe = x0 + ux * pad, x1 - ux * pad
            ax.annotate(
                "", xy=(xe, y1),
                xytext=(xs, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=ec, lw=1.5, mutation_scale=11,
                ),
            )
            ax.text((xs + xe) / 2, y0 + 14,
                    lbl, ha="center", fontsize=6.5, color=ec + "cc",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#04100a", ec="none"))

    # ── START / END diamonds ──────────────────────────────────────────────
    for label, x, y, c in [("START", -20, 220, "#2ecc71"), ("END", 880, 220, "#e74c3c")]:
        ax.add_patch(plt.Polygon(
            [[x, y + 22], [x + 28, y], [x, y - 22], [x - 28, y]],
            closed=True, facecolor=c + "20", edgecolor=c, lw=2, zorder=3,
        ))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=c,
                fontfamily="monospace", zorder=4)

    # START → bmi_agent
    ax.annotate("", xy=(80 - BW/2 - 4, 220), xytext=(8, 220),
                arrowprops=dict(arrowstyle="-|>", color="#2ecc71", lw=1.4, mutation_scale=10))
    # coach_agent → END
    ax.annotate("", xy=(872, 220), xytext=(800 + BW/2 + 4, 220),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.4, mutation_scale=10))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(fc="#2ecc7118", ec="#2ecc71", label="Agent node (ReAct loop)"),
        mpatches.Patch(fc="#3498db40", ec="#3498db", label="Critic node (score_plan tool)"),
        mpatches.Patch(fc="#9b59b640", ec="#9b59b6", label="Tool node"),
        mpatches.Patch(fc="#0000", ec="#3498db", label="Feedback loop edge (dashed)"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              facecolor="#071a0f", edgecolor="#143d22",
              labelcolor="#b8e6c8", fontsize=8)

    # ── ReAct annotation ─────────────────────────────────────────────────
    ax.text(450, 450,
            "Each agent runs a ReAct loop:  THINK (reason about what tool to call)  →  "
            "ACT (invoke tool)  →  OBSERVE (read result)  →  repeat until done",
            ha="center", fontsize=7.5, color="#5a9e72",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#071a0f", ec="#143d22"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#04100a")
    plt.close()
    print(f"[viz] PNG saved → {output_path}")


# ─── Pyvis HTML ──────────────────────────────────────────────────────────────

def save_pyvis(output_path: str = "nutricoach_react_graph.html"):
    try:
        from pyvis.network import Network
    except ImportError:
        print("[viz] pyvis not installed — generating fallback HTML.")
        _fallback_html(output_path)
        return

    net = Network(
        height="680px", width="100%",
        bgcolor="#04100a", font_color="#b8e6c8",
        directed=True, notebook=False,
    )

    # Agent nodes
    for n in NODES:
        c = n["color"]
        net.add_node(
            n["id"],
            label=f"{n['label']}\n{n['tools']}",
            title=n["desc"].replace("\n", "<br/>"),
            color={"background": c + "18", "border": c,
                   "highlight": {"background": c + "35", "border": c}},
            shape="box",
            font={"color": c, "size": 13, "face": "monospace"},
            borderWidth=2,
            x=n["x"] * 1.1, y=n["y"],
            physics=False,
            size=28,
        )

    # Tool nodes (smaller, below agents)
    for t in TOOL_NODES:
        c = t["color"]
        net.add_node(
            t["id"],
            label=t["label"],
            title=f"Tool called by {t['parent']}",
            color={"background": c + "30", "border": c,
                   "highlight": {"background": c + "55", "border": c}},
            shape="ellipse",
            font={"color": c, "size": 10, "face": "monospace"},
            borderWidth=1,
            x=t["x"] * 1.1, y=420,
            physics=False,
            size=14,
        )
        # Dashed edge from agent to tool
        parent_node = next(n for n in NODES if n["id"] == t["parent"])
        net.add_edge(
            t["parent"], t["id"],
            color={"color": t["color"] + "60"},
            dashes=True, width=1,
            arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
        )

    # Agent-to-agent edges
    ec_map = EDGE_COLORS
    for src, dst, lbl, is_loop in EDGES:
        ec = ec_map.get((src, dst), "#2ecc71")
        net.add_edge(
            src, dst,
            label=lbl,
            color={"color": ec, "highlight": ec},
            arrows={"to": {"enabled": True, "scaleFactor": 1.1}},
            font={"color": ec, "size": 9, "face": "monospace", "background": "#04100a"},
            width=2,
            dashes=is_loop,
            smooth={"type": "curvedCCW", "roundness": 0.4} if is_loop else {},
        )

    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {"hover": True, "tooltipDelay": 80, "zoomView": True},
        "edges": {"font": {"strokeWidth": 0}},
    }))

    raw = net.generate_html()

    banner = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');
body{background:#04100a!important;margin:0;}
#mynetwork{border:1px solid #143d22!important;border-radius:0 0 10px 10px;}
</style>
<div style="background:#071a0f;border-bottom:1px solid #143d22;padding:14px 22px;
  display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:#e8f5e9;letter-spacing:-.3px;">
      🌿 NutriCoach ReAct — 5-Agent LangGraph System
    </div>
    <div style="font-size:11px;color:#2d6644;margin-top:2px;font-family:'JetBrains Mono',monospace;">
      Each agent: Think → Act (tool) → Observe · Critic-Revise feedback loop · Conditional routing
    </div>
  </div>
  <div style="display:flex;gap:6px;">
    <span style="font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
      padding:3px 9px;border-radius:3px;border:1px solid rgba(46,204,113,.3);
      background:rgba(46,204,113,.08);color:#2ecc71;">5 ReAct Agents</span>
    <span style="font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
      padding:3px 9px;border-radius:3px;border:1px solid rgba(52,152,219,.3);
      background:rgba(52,152,219,.08);color:#3498db;">Critic-Revise Loop</span>
    <span style="font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
      padding:3px 9px;border-radius:3px;border:1px solid rgba(243,156,18,.3);
      background:rgba(243,156,18,.08);color:#f39c12;">9 Tools</span>
  </div>
</div>
<div style="background:#071a0f;border-bottom:1px solid #143d22;padding:6px 22px;
  font-size:10px;color:#5a9e72;font-family:'JetBrains Mono',monospace;">
  Hover nodes for details · Scroll to zoom · Drag to pan · Dashed edges = ReAct tool calls or feedback loops
</div>"""

    raw = raw.replace("<body>", "<body>" + banner, 1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"[viz] Interactive HTML saved → {output_path}")


def _fallback_html(output_path: str):
    rows = "\n".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        f'<span style="border:1px solid {EDGE_COLORS.get((s,d),"#2ecc71")};color:{EDGE_COLORS.get((s,d),"#2ecc71")};'
        f'padding:3px 10px;border-radius:4px;font-size:12px;">{s}</span>'
        f'<span style="color:#2d6644;">——{lbl}——▶</span>'
        f'<span style="border:1px solid #143d22;color:#5a9e72;padding:3px 10px;border-radius:4px;font-size:12px;">{d}</span>'
        f'</div>'
        for s, d, lbl, _ in EDGES
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
<title>NutriCoach ReAct Graph</title>
<style>body{{background:#04100a;color:#b8e6c8;font-family:monospace;padding:24px;}}
h2{{color:#2ecc71;}}p{{color:#2d6644;font-size:12px;}}</style></head>
<body><h2>🌿 NutriCoach ReAct Agent Graph</h2>
<p>5-agent system (install pyvis for interactive version)</p>
{rows}</body></html>""")
    print(f"[viz] Fallback HTML saved → {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize NutriCoach ReAct agent graph.")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--no-open", action="store_true", help="Skip auto-opening browser")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    html_path = os.path.join(args.out_dir, "nutricoach_react_graph.html")
    png_path  = os.path.join(args.out_dir, "nutricoach_react_graph.png")

    print("\n=== NutriCoach ReAct Graph Visualizer ===\n")
    save_matplotlib(png_path)
    save_pyvis(html_path)
    print("\nDone.")

    if not args.no_open:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path))
        except Exception:
            pass
