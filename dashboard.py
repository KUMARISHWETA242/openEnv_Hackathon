#!/usr/bin/env python3
import html
import json
import os
from typing import Dict, List

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import plotly.graph_objects as go

from satellite import (
    EasyTask,
    HardTask,
    MediumTask,
    SatelliteAction,
    SatelliteTaskEnv,
)


VALID_ACTIONS = ["capture", "downlink", "maintain", "idle"]
TASKS = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}
MAX_VISIBLE_SATELLITES = 8


def create_session(task_name: str = "easy") -> Dict:
    task = TASKS[task_name]()
    env = SatelliteTaskEnv(task_name=task_name)
    observation = env.reset()
    return {
        "env": env,
        "task_name": task_name,
        "task_description": task.description,
        "observation": observation,
        "done": False,
        "history": [],
    }


def heuristic_policy(observation) -> Dict[int, str]:
    actions: Dict[int, str] = {}
    has_capture_task = any(task.get("type") == "image_capture" for task in observation.pending_tasks)
    has_downlink_task = any(task.get("type") == "data_downlink" for task in observation.pending_tasks)

    for sat in observation.satellites:
        if sat.battery <= 15:
            actions[sat.id] = "maintain"
        elif sat.storage >= 80 and has_downlink_task:
            actions[sat.id] = "downlink"
        elif has_capture_task and sat.battery > 20 and sat.storage < 80:
            actions[sat.id] = "capture"
        elif sat.storage >= 50 and has_downlink_task and sat.battery > 10:
            actions[sat.id] = "downlink"
        elif sat.battery < 35:
            actions[sat.id] = "maintain"
        else:
            actions[sat.id] = "idle"

    return actions


def random_policy(observation) -> Dict[int, str]:
    import random

    return {sat.id: random.choice(VALID_ACTIONS) for sat in observation.satellites}


def manual_policy(observation, manual_actions: List[str]) -> Dict[int, str]:
    actions: Dict[int, str] = {}
    for sat in observation.satellites:
        chosen = manual_actions[sat.id] if sat.id < len(manual_actions) else "idle"
        actions[sat.id] = chosen if chosen in VALID_ACTIONS else "idle"
    return actions


def build_metrics_markdown(session: Dict) -> str:
    observation = session["observation"]
    history = session["history"]
    total_reward = sum(item["reward"] for item in history)
    avg_battery = sum(sat.battery for sat in observation.satellites) / max(1, len(observation.satellites))
    avg_storage = sum(sat.storage for sat in observation.satellites) / max(1, len(observation.satellites))
    captures = sum(1 for item in history for action in item["actions"].values() if action == "capture")
    downlinks = sum(1 for item in history for action in item["actions"].values() if action == "downlink")
    maintains = sum(1 for item in history for action in item["actions"].values() if action == "maintain")
    status = "done" if session["done"] else "running"
    return (
        "### Mission Status\n"
        f"- Task: `{session['task_name']}`\n"
        f"- Description: {session['task_description']}\n"
        f"- Time step: `{observation.time_step}` / `{session['env'].state().max_steps}`\n"
        f"- Episode status: `{status}`\n"
        f"- Total reward: `{total_reward:.2f}`\n"
        f"- Avg battery: `{avg_battery:.1f}`\n"
        f"- Avg storage: `{avg_storage:.1f}`\n"
        f"- Captures: `{captures}` | Downlinks: `{downlinks}` | Maintains: `{maintains}`"
    )


def progress_bar(value: float, color: str) -> str:
    width = max(0.0, min(100.0, value))
    return (
        "<div style='background:#e5e7eb;border-radius:999px;height:10px;overflow:hidden;'>"
        f"<div style='width:{width}%;background:{color};height:10px;'></div>"
        "</div>"
    )


def panel_wrap(content: str) -> str:
    return (
        "<div style='color:#111827;background:#ffffff;'>"
        + content
        + "</div>"
    )


def build_satellite_cards(observation, large: bool = False) -> str:
    min_width = "280px" if large else "220px"
    pad = "18px" if large else "14px"
    title_size = "16px" if large else "14px"
    meta_size = "13px" if large else "12px"
    cards = []
    for sat in observation.satellites:
        cards.append(
            f"""
            <div style="border:1px solid #d1d5db;border-radius:16px;padding:{pad};background:#ffffff;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <strong style="font-size:{title_size};">Satellite {sat.id}</strong>
                <span style="font-size:{meta_size};color:#4b5563;">last: {html.escape(str(sat.last_action))}</span>
              </div>
              <div style="font-size:13px;color:#374151;margin-bottom:6px;">Battery {sat.battery:.1f}%</div>
              {progress_bar(sat.battery, "#2563eb")}
              <div style="font-size:13px;color:#374151;margin:10px 0 6px;">Storage {sat.storage:.1f}%</div>
              {progress_bar(sat.storage, "#f97316")}
              <div style="margin-top:10px;font-size:12px;color:#6b7280;">
                lat={sat.position[0]:.1f}, lon={sat.position[1]:.1f}, alt={sat.position[2]:.1f} km
              </div>
            </div>
            """
        )
    return panel_wrap(
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax({min_width},1fr));gap:12px;'>"
        + "".join(cards)
        + "</div>"
    )


def build_task_panel(observation) -> str:
    tasks = "".join(
        f"<li><code>{html.escape(json.dumps(task))}</code></li>"
        for task in observation.pending_tasks
    ) or "<li>None</li>"
    weather = "".join(
        f"<li><strong>{html.escape(region)}</strong>: {cover:.2f}</li>"
        for region, cover in observation.weather_conditions.items()
    ) or "<li>None</li>"
    stations = "".join(
        f"<li>Station {idx}: <code>{html.escape(str(station))}</code></li>"
        for idx, station in enumerate(observation.ground_stations)
    ) or "<li>None</li>"

    return panel_wrap(
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;'>"
        f"<div style='border:1px solid #d1d5db;border-radius:16px;padding:14px;'><h3>Pending Tasks</h3><ul>{tasks}</ul></div>"
        f"<div style='border:1px solid #d1d5db;border-radius:16px;padding:14px;'><h3>Weather</h3><ul>{weather}</ul></div>"
        f"<div style='border:1px solid #d1d5db;border-radius:16px;padding:14px;'><h3>Ground Stations</h3><ul>{stations}</ul></div>"
        "</div>"
    )


def build_latest_action_panel(history: List[Dict]) -> str:
    if not history:
        return panel_wrap("<div style='border:1px solid #d1d5db;border-radius:16px;padding:14px;'>No actions taken yet.</div>")

    latest = history[-1]
    actions = "".join(
        f"<li>Satellite <strong>{sat}</strong> -> <code>{html.escape(act)}</code></li>"
        for sat, act in latest["actions"].items()
    )
    components = "".join(
        f"<li><code>{html.escape(name)}</code>: {value:.1f}</li>"
        for name, value in latest["components"].items()
    ) or "<li>none</li>"
    return panel_wrap(
        "<div style='border:1px solid #d1d5db;border-radius:16px;padding:14px;background:#ffffff;'>"
        f"<h3>Latest Step</h3><p><strong>Step:</strong> {latest['step']}<br><strong>Reward:</strong> {latest['reward']:.2f}</p>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'><div><strong>Actions</strong><ul>{actions}</ul></div><div><strong>Reward Components</strong><ul>{components}</ul></div></div>"
        "</div>"
    )


def build_history_table(history: List[Dict]) -> str:
    rows = []
    for entry in history[-20:]:
        rows.append(
            "<tr>"
            f"<td>{entry['step']}</td>"
            f"<td>{entry['reward']:.2f}</td>"
            f"<td>{html.escape(', '.join(f'S{sat}:{act}' for sat, act in entry['actions'].items()))}</td>"
            f"<td>{html.escape(', '.join(f'{k}={v:.1f}' for k, v in entry['components'].items()) or '-')}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4'>No history yet.</td></tr>")

    return panel_wrap(
        "<table style='width:100%;border-collapse:collapse;background:#ffffff;'>"
        "<thead><tr style='background:#f3f4f6;'><th style='text-align:left;padding:8px;'>Step</th><th style='text-align:left;padding:8px;'>Reward</th><th style='text-align:left;padding:8px;'>Actions</th><th style='text-align:left;padding:8px;'>Components</th></tr></thead>"
        "<tbody>"
        + "".join(f"<tr style='border-top:1px solid #e5e7eb;'>{row[4:]}" if row.startswith("<tr>") else row for row in rows)
        + "</tbody></table>"
    )


def build_reward_plot(history: List[Dict]) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(
            title="Reward Timeline",
            template="plotly_white",
            annotations=[dict(text="No reward data yet", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
        )
        return fig

    steps = [entry["step"] for entry in history]
    rewards = [entry["reward"] for entry in history]
    cumulative = []
    total = 0.0
    for reward in rewards:
        total += reward
        cumulative.append(total)

    fig.add_trace(go.Bar(x=steps, y=rewards, name="Step reward", marker_color="rgba(20,184,166,0.75)"))
    fig.add_trace(go.Scatter(x=steps, y=cumulative, mode="lines+markers", name="Cumulative reward", line=dict(color="#111827", width=3)))
    fig.update_layout(
        title="Reward Timeline",
        template="plotly_white",
        height=340,
        margin=dict(l=30, r=20, t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0.01),
    )
    fig.update_xaxes(title="Step")
    fig.update_yaxes(title="Reward")
    return fig


def _satellite_marker_opacity(battery: float, storage: float) -> float:
    health = max(0.0, min(1.0, battery / 100.0))
    free_capacity = max(0.0, min(1.0, (100.0 - storage) / 100.0))
    return 0.25 + 0.75 * ((health + free_capacity) / 2.0)


def build_ground_track_plot(session: Dict) -> go.Figure:
    observation = session["observation"]
    history = session["history"]
    fig = go.Figure()

    for idx, station in enumerate(observation.ground_stations):
        fig.add_trace(
            go.Scatter(
                x=[station[1]],
                y=[station[0]],
                mode="markers+text",
                name=f"GS{idx}",
                text=[f"GS{idx}"],
                textposition="top center",
                marker=dict(symbol="triangle-up", size=14, color="#dc2626"),
                hovertemplate=f"Ground Station {idx}<br>Lat: {station[0]:.2f}<br>Lon: {station[1]:.2f}<extra></extra>",
            )
        )

    for sat in observation.satellites:
        path_points = []
        for entry in history:
            if "positions" in entry and sat.id in entry["positions"]:
                path_points.append(entry["positions"][sat.id])
        current_lat, current_lon, current_alt = sat.position
        if not path_points or path_points[-1] != sat.position:
            path_points.append(sat.position)

        lats = [point[0] for point in path_points]
        lons = [point[1] for point in path_points]
        hover_text = (
            f"Satellite {sat.id}<br>"
            f"Lat: {current_lat:.2f}<br>"
            f"Lon: {current_lon:.2f}<br>"
            f"Alt: {current_alt:.1f} km<br>"
            f"Battery: {sat.battery:.1f}%<br>"
            f"Storage: {sat.storage:.1f}%<br>"
            f"Last action: {sat.last_action}"
        )
        opacity = _satellite_marker_opacity(sat.battery, sat.storage)

        if len(path_points) > 1:
            fig.add_trace(
                go.Scatter(
                    x=lons,
                    y=lats,
                    mode="lines",
                    name=f"S{sat.id} path",
                    line=dict(width=2),
                    opacity=max(0.25, opacity - 0.15),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[current_lon],
                y=[current_lat],
                mode="markers+text",
                name=f"S{sat.id}",
                text=[f"S{sat.id}"],
                textposition="top right",
                marker=dict(
                    size=14 + (sat.storage * 0.12),
                    color=f"rgba(37,99,235,{opacity:.3f})",
                    line=dict(color="#1d4ed8", width=1),
                ),
                hovertemplate=hover_text + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Ground Track With Satellite Paths",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="closest",
        legend=dict(orientation="h", y=1.08, x=0.01),
    )
    fig.update_xaxes(title="Longitude", range=[-180, 180], dtick=60, zeroline=True)
    fig.update_yaxes(title="Latitude", range=[-90, 90], dtick=30, zeroline=True)
    return fig


def get_manual_updates(observation) -> List[gr.update]:
    updates: List[gr.update] = []
    for idx in range(MAX_VISIBLE_SATELLITES):
        if idx < len(observation.satellites):
            updates.append(gr.update(visible=True, label=f"Satellite {idx} action", value="idle"))
        else:
            updates.append(gr.update(visible=False))
    return updates


def render_session(session: Dict):
    observation = session["observation"]
    done_text = "Simulation finished." if session["done"] else "Simulation active."
    ground_track_plot = build_ground_track_plot(session)
    return (
        session,
        build_metrics_markdown(session),
        build_latest_action_panel(session["history"]),
        build_task_panel(observation),
        build_satellite_cards(observation),
        build_satellite_cards(observation, large=True),
        build_history_table(session["history"]),
        build_reward_plot(session["history"]),
        ground_track_plot,
        ground_track_plot,
        done_text,
        *get_manual_updates(observation),
    )


def reset_simulation(task_name: str):
    return render_session(create_session(task_name))


def resolve_actions(session: Dict, policy_name: str, manual_actions: List[str]) -> Dict[int, str]:
    observation = session["observation"]
    if policy_name == "manual":
        return manual_policy(observation, manual_actions)
    if policy_name == "random":
        return random_policy(observation)
    return heuristic_policy(observation)


def execute_steps(session: Dict, task_name: str, policy_name: str, run_steps: int, *manual_actions):
    if not session or session.get("task_name") != task_name:
        session = create_session(task_name)

    run_steps = max(1, int(run_steps))

    for _ in range(run_steps):
        if session["done"]:
            break
        chosen_actions = resolve_actions(session, policy_name, list(manual_actions))
        action = SatelliteAction(satellite_actions=chosen_actions)
        observation, reward, done, _ = session["env"].step(action)
        session["observation"] = observation
        session["done"] = done
        session["history"].append(
            {
                "step": observation.time_step,
                "actions": chosen_actions,
                "reward": reward.value,
                "components": reward.components,
                "positions": {sat.id: sat.position for sat in observation.satellites},
            }
        )

    return render_session(session)


def build_dashboard() -> gr.Blocks:
    css = """
    .gradio-container {max-width: 1400px !important; margin: 0 auto !important;}
    .gradio-container, .gradio-container * {color: #111827;}
    .gradio-container .prose, .gradio-container .prose * {color: #111827 !important;}
    #topbar .gap {gap: 12px !important;}
    #control-card, #summary-card, #panel-card {
        border: 1px solid #d1d5db;
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    #control-card, #summary-card {padding: 10px;}
    #panel-card {padding: 6px 10px 10px 10px;}
    .compact-markdown p {margin: 0.2rem 0 !important;}
    .compact-markdown ul {margin-top: 0.2rem !important; margin-bottom: 0.2rem !important;}
    .compact-markdown code, #panel-card code {
        background: #f3f4f6;
        color: #111827 !important;
        padding: 1px 5px;
        border-radius: 6px;
    }
    .gr-box, .gr-panel {background: #ffffff !important;}
    """

    with gr.Blocks(title="Satellite Simulation Dashboard", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            """
            # Satellite Simulation Dashboard
            Run the constellation environment, compare policies, and monitor state, actions, rewards, weather, and positions in a compact view.
            """
        )

        session_state = gr.State(create_session("easy"))

        with gr.Row(elem_id="topbar"):
            with gr.Column(scale=4, elem_id="control-card"):
                with gr.Row():
                    task_name = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task", scale=1)
                    policy_name = gr.Radio(choices=["heuristic", "random", "manual"], value="heuristic", label="Policy", scale=2)
                    run_steps = gr.Slider(minimum=1, maximum=20, step=1, value=1, label="Steps", scale=2)
                with gr.Row():
                    step_btn = gr.Button("Run Step(s)", variant="primary")
                    reset_btn = gr.Button("Reset Simulation", variant="secondary")
                    status_text = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("Manual Actions", open=False):
                    manual_dropdowns = []
                    with gr.Row():
                        for idx in range(4):
                            manual_dropdowns.append(
                                gr.Dropdown(
                                    choices=VALID_ACTIONS,
                                    value="idle",
                                    label=f"Satellite {idx}",
                                    visible=idx < 3,
                                )
                            )
                    with gr.Row():
                        for idx in range(4, MAX_VISIBLE_SATELLITES):
                            manual_dropdowns.append(
                                gr.Dropdown(
                                    choices=VALID_ACTIONS,
                                    value="idle",
                                    label=f"Satellite {idx}",
                                    visible=False,
                                )
                            )

            with gr.Column(scale=5, elem_id="summary-card"):
                metrics_md = gr.Markdown(elem_classes=["compact-markdown"])
                latest_html = gr.HTML()

        with gr.Row():
            with gr.Column(scale=7, elem_id="panel-card"):
                satellite_html = gr.HTML()
            with gr.Column(scale=5, elem_id="panel-card"):
                position_plot = gr.Plot()

        with gr.Row():
            with gr.Column(scale=5, elem_id="panel-card"):
                task_html = gr.HTML()
            with gr.Column(scale=7, elem_id="panel-card"):
                reward_plot = gr.Plot()

        with gr.Row():
            with gr.Column(elem_id="panel-card"):
                history_html = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Expanded Fleet"):
                with gr.Column(elem_id="panel-card"):
                    satellite_large_html = gr.HTML()
                with gr.Column(elem_id="panel-card"):
                    position_plot_large = gr.Plot()

        reset_outputs = [
            session_state,
            metrics_md,
            latest_html,
            task_html,
            satellite_html,
            satellite_large_html,
            history_html,
            reward_plot,
            position_plot,
            position_plot_large,
            status_text,
            *manual_dropdowns,
        ]

        reset_btn.click(fn=reset_simulation, inputs=[task_name], outputs=reset_outputs)
        task_name.change(fn=reset_simulation, inputs=[task_name], outputs=reset_outputs)
        step_btn.click(
            fn=execute_steps,
            inputs=[session_state, task_name, policy_name, run_steps, *manual_dropdowns],
            outputs=reset_outputs,
        )
        demo.load(fn=lambda: render_session(create_session("easy")), outputs=reset_outputs)

    return demo


if __name__ == "__main__":
    build_dashboard().launch()
