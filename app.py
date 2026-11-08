"""Interactive Streamlit interface for TripMates Lite."""

from __future__ import annotations

from dataclasses import asdict
from datetime import time
from html import escape
from pathlib import Path

from ga_core import GAConfig

ROOT = Path(__file__).resolve().parent
POI_FILE = ROOT / "poi_data_calculated_final.csv"
MATRIX_FILE = ROOT / "walking_time_matrix_final.csv"
HISTORY_FILE = ROOT / "data" / "itinerary_history.sqlite3"
WEATHER_CACHE_FILE = ROOT / "data" / "weather_profile.json"
KUALA_LUMPUR = {"latitude": 3.1390, "longitude": 101.6869}

# The thesis benchmark remains the default in the GA modules.  These smaller
# populations are used only by the interactive UI so a visitor can explore a
# route without waiting for a full experimental run.
QUICK_INITIAL_CONFIG = GAConfig(
    population_size=12,
    maximum_generations=12,
    stagnation_limit=4,
)
QUICK_REPLANNING_CONFIG = GAConfig(
    population_size=12,
    maximum_generations=12,
    stagnation_limit=4,
)


def time_to_minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def weather_emoji(category: str) -> str:
    return {"rain": "🌧️", "hot": "☀️", "normal": "⛅"}.get(category, "🌤️")


def weather_label(code: int | None) -> str:
    return {
        0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
        51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
        61: "Light rain", 63: "Rain", 65: "Heavy rain",
        80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
        95: "Thunderstorm",
    }.get(code, "Current conditions")


def inject_styles(st) -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f8f6; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; }
        [data-testid="stSidebar"] { background: #102a24; }
        [data-testid="stSidebar"] * { color: #f4fbf7; }
        .hero { background: linear-gradient(120deg,#0d3b31,#126d5b 60%,#23a17f);
          border-radius:22px;color:#fff;padding:2rem 2.2rem;margin-bottom:1.4rem;
          box-shadow:0 12px 28px rgba(14,65,53,.18); }
        .hero h1 { margin:0 0 .35rem;font-size:2.2rem; }
        .hero p { margin:0;opacity:.9;font-size:1.05rem; }
        .eyebrow { color:#8fe4c8;font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase; }
        .route-stop { background:#fff;border-left:4px solid #23a17f;border-radius:0 12px 12px 0;
          padding:.75rem 1rem;margin:.45rem 0;box-shadow:0 3px 11px rgba(18,61,48,.07); }
        .route-time { color:#16735e;font-size:.84rem;font-weight:700; }
        .route-name { color:#183c33;font-weight:700;margin-top:.1rem; }
        .route-detail { color:#5f736d;font-size:.86rem; }
        .spot-card { background:#fff;border:1px solid #d9ebe3;border-radius:16px;
          padding:1.1rem 1.2rem;box-shadow:0 5px 16px rgba(18,61,48,.08); }
        .spot-category { color:#168269;font-size:.78rem;font-weight:700;
          letter-spacing:.08em;text-transform:uppercase; }
        .spot-card h3 { color:#183c33;margin:.25rem 0 .75rem; }
        .spot-fact { color:#526b63;margin:.38rem 0;font-size:.92rem; }
        div[data-testid="stMetric"] { background:#fff;border-radius:14px;padding:.6rem .8rem;
          box-shadow:0 3px 12px rgba(18,61,48,.06); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def route_map_data(schedule, pois, count: int):
    entries = schedule.entries[:count]
    points = [
        {
            "position": [pois[item.poi_id].longitude, pois[item.poi_id].latitude],
            "name": item.poi_name, "stop": item.sequence_index + 1,
            "label": str(item.sequence_index + 1),
            "arrival": item.arrival_clock, "leave": item.visit_end_clock,
            "category": pois[item.poi_id].category,
            "duration": f"{pois[item.poi_id].visit_duration_min:.0f} min",
            "cost": f"RM {pois[item.poi_id].admission_cost_rm:.2f}",
            "utility": f"{pois[item.poi_id].utility_score:.2f}",
            "color": [246, 169, 42, 255] if item == entries[-1] else [13, 87, 71, 235],
            "radius": 115 if item == entries[-1] else 78,
        }
        for item in entries
    ]
    path = [point["position"] for point in points]
    return points, [{"path": path}] if len(path) > 1 else []


def render_route_map(st, schedule, pois, *, key: str) -> None:
    import pydeck as pdk

    st.markdown("#### Route explorer")
    count = st.slider(
        "Follow the connected route",
        1, len(schedule.entries), len(schedule.entries), key=f"route_{key}",
        help="Move this slider to reveal the walking route stop by stop.",
    )
    points, paths = route_map_data(schedule, pois, count)
    current = points[-1]
    layers = []
    if paths:
        layers.append(pdk.Layer(
            "PathLayer", data=paths, get_path="path", get_color=[35, 161, 127, 210],
            width_scale=5, width_min_pixels=4,
        ))
    layers += [
        pdk.Layer(
            "ScatterplotLayer", data=points, get_position="position",
            get_fill_color="color", get_radius="radius", radius_min_pixels=7, pickable=True,
        ),
        pdk.Layer(
            "TextLayer", data=points, get_position="position", get_text="label",
            get_color=[255, 255, 255, 255], get_size=15,
            get_alignment_baseline="center", get_text_anchor="middle",
        ),
    ]
    map_col, detail_col = st.columns([2.4, 1])
    with map_col:
        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=current["position"][1], longitude=current["position"][0],
                    zoom=14, pitch=42,
                ),
                tooltip={
                    "html": (
                        "<b>Stop {stop}: {name}</b><br/>{category}<br/>"
                        "Arrive {arrival} · leave {leave}<br/>"
                        "Visit {duration} · {cost}<br/>Utility {utility}"
                    ),
                    "style": {"backgroundColor": "#0d3b31", "color": "white"},
                },
                map_style="light",
            ),
            height=440, width="stretch",
        )
    selected_entry = schedule.entries[count - 1]
    selected_poi = pois[selected_entry.poi_id]
    outdoor_percent = max(
        0.0,
        100.0 * (1.0 - selected_poi.indoor_ratio - selected_poi.shelter_ratio),
    )
    with detail_col:
        st.markdown(
            f"""<div class="spot-card">
            <div class="spot-category">Stop {count} · {escape(selected_poi.category)}</div>
            <h3>{escape(selected_poi.name)}</h3>
            <div class="spot-fact">🕐 Arrive {escape(selected_entry.arrival_clock)} · leave {escape(selected_entry.visit_end_clock)}</div>
            <div class="spot-fact">🚶 {selected_entry.travel_min:.0f} min walk from previous stop</div>
            <div class="spot-fact">📍 {selected_poi.visit_duration_min:.0f} min suggested visit</div>
            <div class="spot-fact">🎟️ RM {selected_poi.admission_cost_rm:.2f} admission</div>
            <div class="spot-fact">✨ Utility score {selected_poi.utility_score:.2f}</div>
            <div class="spot-fact">🌤️ {outdoor_percent:.0f}% estimated outdoor exposure</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_schedule(st, schedule, pois, *, title: str, key: str) -> None:
    st.subheader(title)
    render_route_map(st, schedule, pois, key=key)
    st.markdown("#### At a glance")
    timeline_columns = st.columns(2)
    for index, item in enumerate(schedule.entries):
        with timeline_columns[index % 2]:
            st.markdown(
                f"""<div class="route-stop">
                <div class="route-time">{escape(item.arrival_clock)} → {escape(item.visit_end_clock)}</div>
                <div class="route-name">{item.sequence_index + 1}. {escape(item.poi_name)}</div>
                <div class="route-detail">Walk {item.travel_min:.0f} min · Visit {item.visit_duration_min:.0f} min · RM {item.admission_cost_rm:.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    st.markdown("#### Trip metrics")
    metrics = st.columns(4)
    metrics[0].metric("Stops", len(schedule.entries))
    metrics[1].metric("Walking", f"{schedule.total_travel_min:.0f} min")
    metrics[2].metric("Finish", schedule.finish_clock)
    metrics[3].metric("Admission", f"RM {schedule.total_cost_rm:.2f}")


def minutes_to_clock(value) -> str:
    if value is None:
        return "—"
    hour, minute = divmod(int(round(float(value))), 60)
    return f"{hour % 24:02d}:{minute:02d}"


def render_saved_summary(st, stored, pois) -> None:
    """Present saved inputs and measurements without exposing raw JSON."""

    parameters = dict(stored.record.request_parameters)
    st.markdown("#### Trip setup")
    setup_columns = st.columns(4)
    setup_columns[0].metric("Trip date", parameters.get("trip_date", "Not recorded"))
    setup_columns[1].metric("Start", minutes_to_clock(parameters.get("start_time_min")))
    setup_columns[2].metric("Finish by", minutes_to_clock(parameters.get("end_time_min")))
    budget = parameters.get("cost_budget_rm")
    setup_columns[3].metric("Budget", f"RM {float(budget):.2f}" if budget is not None else "—")

    mandatory_names = [
        pois[poi_id].name
        for poi_id in parameters.get("mandatory_poi_ids", [])
        if poi_id in pois
    ]
    preferred_names = [
        pois[poi_id].name
        for poi_id in parameters.get("preferred_poi_ids", [])
        if poi_id in pois
    ]
    st.write("**Must visit:** " + (", ".join(mandatory_names) or "Starting attraction only"))
    st.write("**Preferred:** " + (", ".join(preferred_names) or "No preference supplied"))

    weather = parameters.get("weather", {})
    if weather:
        temperature = weather.get("temperature_c")
        temperature_text = f"{float(temperature):.0f}°C" if temperature is not None else "Unavailable"
        st.info(
            f"Weather used: {str(weather.get('category', 'normal')).title()} · "
            f"{temperature_text} · {weather.get('source_type', 'unknown')} source · "
            f"comfort multiplier {float(weather.get('thermal_multiplier', 1.0)):.1f}×"
        )

    st.markdown("#### Saved results")
    result_labels = {
        "achieved_utility": "Achieved utility",
        "weather_weighted_exposure_score": "Weather exposure score",
        "total_outdoor_exposure_min": "Outdoor exposure (min)",
        "total_waiting_min": "Waiting (min)",
        "total_visit_min": "Visiting (min)",
        "replanning_latency_ms": "Replanning (ms)",
        "generations_executed": "GA generations",
    }
    visible_metrics = [
        (result_labels[name], value)
        for name, value in stored.metrics.items()
        if name in result_labels
    ]
    metric_columns = st.columns(4)
    for index, (label, value) in enumerate(visible_metrics):
        metric_columns[index % 4].metric(label, f"{value:.2f}")


def render_saved_view(st, history, service) -> None:
    st.subheader("Saved itineraries")
    st.caption("Open a saved trip with its complete map, timeline, inputs, and results.")
    records = history.list_runs()
    if not records:
        st.info("No saved itineraries yet. Generate a trip, name it, and save it first.")
        return

    labels = {
        f"{record.itinerary_name} · {record.created_at[:10]}": record.run_id
        for record in records
    }
    selected_label = st.selectbox("Choose an itinerary", list(labels))
    stored = history.load(labels[selected_label])
    if stored is None:
        st.error("This saved itinerary could not be loaded.")
        return

    st.caption(
        f"Saved {stored.record.created_at[:16].replace('T', ' ')} · "
        f"{stored.record.planner_id.replace('_', ' ').title()} · "
        f"{stored.record.configuration_id.replace('_v1', '').replace('_', ' ').title()}"
    )
    render_schedule(
        st,
        stored.schedule,
        service.data.pois,
        title=stored.record.itinerary_name,
        key=f"saved_{stored.record.run_id}",
    )
    render_saved_summary(st, stored, service.data.pois)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as error:
        raise RuntimeError("Install requirements before running the app.") from error

    from itinerary_history import ItineraryHistoryRepository, ItineraryRunRecord
    from itinerary_service import ItineraryService, RequestValidationError, TripRequest
    from weather_service import WeatherServiceAdapter, predefined_weather_profile

    st.set_page_config(page_title="TripMates Lite", page_icon="🧭", layout="wide")
    inject_styles(st)

    @st.cache_resource
    def load_service():
        return ItineraryService.from_files(poi_file=POI_FILE, matrix_file=MATRIX_FILE)

    service = load_service()
    history = ItineraryHistoryRepository(HISTORY_FILE)
    weather = WeatherServiceAdapter(cache_file=WEATHER_CACHE_FILE)
    if "weather_profile" not in st.session_state:
        st.session_state["weather_profile"] = weather.get_profile(
            **KUALA_LUMPUR, applicable_at="interactive",
            fallback_profile=predefined_weather_profile(),
        )
    profile = st.session_state["weather_profile"]
    poi_options = {f"{poi.poi_id} - {poi.name}": poi.poi_id for poi in service.data.pois.values()}

    st.markdown(
        """<section class="hero"><div class="eyebrow">Kuala Lumpur itinerary companion</div>
        <h1>TripMates Lite 🧭</h1>
        <p>Build a walkable day, explore it on a connected map, and adapt it when conditions change.</p>
        </section>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.header("TripMates Lite 🧭")
        navigation = st.radio(
            "Navigation",
            ["Plan a trip", "Saved itineraries"],
            label_visibility="collapsed",
        )
        st.divider()
        st.header("Live Kuala Lumpur weather")
        profile = st.session_state["weather_profile"]
        temperature = f"{profile.temperature_c:.0f}°C" if profile.temperature_c is not None else "Unavailable"
        st.markdown(f"### {weather_emoji(profile.category)} {temperature}\n{weather_label(profile.weather_code)}")
        st.caption(f"Open-Meteo · {profile.source_type} · comfort multiplier {profile.thermal_multiplier:.1f}×")
        if profile.fallback_reason:
            st.caption("Using saved or predefined conditions because live weather was unavailable.")
        if st.button("Refresh weather", width="stretch"):
            st.session_state["weather_profile"] = weather.get_profile(
                **KUALA_LUMPUR, applicable_at="interactive",
                fallback_profile=predefined_weather_profile(),
            )
            st.rerun()

    if navigation == "Saved itineraries":
        render_saved_view(st, history, service)
        return

    st.subheader("Plan a trip")
    if profile.category == "rain":
        st.warning(
            "Live rain detected. Automatic weather mode will use the thermal-aware Heavy Rain response."
        )
    elif profile.category == "hot":
        st.warning(
            "High heat detected. Automatic weather mode will use the thermal-aware Heatwave response."
        )
    else:
        st.info(
            "Live conditions are normal. Automatic weather mode will keep the initial itinerary."
        )

    with st.form("trip_input"):
        left, right = st.columns(2)
        with left:
            trip_date = st.date_input("Trip date")
            start_label = st.selectbox("Starting attraction", list(poi_options))
            start_col, end_col = st.columns(2)
            start_time = start_col.time_input("Start", value=time(9, 0))
            end_time = end_col.time_input("Finish by", value=time(17, 0))
            budget = st.number_input("Admission budget (RM)", 0.0, value=160.0, step=10.0)
        with right:
            mandatory = st.multiselect(
                "Must-visit attractions", list(poi_options),
                help="Your starting attraction is included automatically.",
            )
            preferred = st.multiselect(
                "Preferred attractions",
                list(poi_options),
                help="Preferred attractions receive a 25% soft utility boost; they are not mandatory.",
            )
            scenario_choice = st.selectbox(
                "Planning conditions",
                ["Use live weather automatically", "Normal conditions", "S1 Heavy Rain", "S2 Movement Delay", "S3 Closure",
                 "S4 Congestion", "S5 Heatwave", "S6 Combined"],
                help="Use the current Open-Meteo conditions, or choose a scenario to demonstrate.",
            )
            trigger_variant = st.selectbox("When it happens", ["nominal", "early", "late"])
            planning_mode = st.radio(
                "Planning speed",
                ["Quick preview", "Research benchmark"],
                horizontal=True,
                help=(
                    "Quick preview uses a smaller GA population for responsive "
                    "exploration. Research benchmark uses the thesis configuration."
                ),
            )
        submitted = st.form_submit_button("Create my itinerary", type="primary", width="stretch")

    if submitted:
        request = TripRequest(
            trip_date=trip_date, start_poi_id=poi_options[start_label],
            start_time_min=time_to_minutes(start_time), end_time_min=time_to_minutes(end_time),
            cost_budget_rm=float(budget),
            mandatory_poi_ids=tuple(poi_options[label] for label in mandatory),
            preferred_poi_ids=tuple(poi_options[label] for label in preferred),
        )
        try:
            instance = service.build_instance(request)
        except RequestValidationError as error:
            for issue in error.issues:
                st.error(f"{issue.field}: {issue.message}")
        else:
            initial_config = (
                QUICK_INITIAL_CONFIG
                if planning_mode == "Quick preview"
                else None
            )
            replanning_config = (
                QUICK_REPLANNING_CONFIG
                if planning_mode == "Quick preview"
                else None
            )
            with st.spinner("Finding a walkable itinerary..."):
                initial = service.plan_initial_itinerary(
                    instance,
                    config=initial_config,
                )
            st.session_state["initial"] = initial
            st.session_state["replan"] = None
            st.session_state["planning_instance"] = instance
            st.session_state["planning_mode"] = planning_mode
            st.session_state["weather_mode"] = scenario_choice
            if initial.status != "success":
                st.error(initial.message)
            else:
                scenario_id = None
                if scenario_choice == "Use live weather automatically":
                    if profile.category == "rain":
                        scenario_id = "S1"
                    elif profile.category == "hot":
                        scenario_id = "S5"
                elif scenario_choice != "Normal conditions":
                    scenario_id = scenario_choice.split()[0]

            if initial.status == "success" and scenario_id is not None:
                scenario = next(
                    item for item in service.scenario_instances_for(instance)
                    if item.instance_id == "{}_{}".format(scenario_id, trigger_variant)
                )
                with st.spinner("Adapting the unfinished route..."):
                    st.session_state["replan"] = service.replan(
                        instance=instance, initial_outcome=initial,
                        scenario_instance=scenario, planner_id="tripmates_lite",
                        config=replanning_config,
                    )

    initial = st.session_state.get("initial")
    replan = st.session_state.get("replan")
    if initial is not None and initial.schedule is not None:
        render_schedule(st, initial.schedule, service.data.pois, title="Your initial itinerary", key="initial")

    if replan is not None:
        st.divider()
        if replan.status == "success":
            st.success(replan.explanation)
            render_schedule(st, replan.repaired_schedule, service.data.pois, title="Your adapted itinerary", key="adapted")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Weather exposure score", f"{replan.exposure.weather_weighted_exposure_score:.1f}")
            metric_columns[1].metric("Outdoor exposure", f"{replan.exposure.total_outdoor_exposure_min:.0f} min")
            metric_columns[2].metric("Replanning time", f"{replan.replanning_latency_ms:.0f} ms")
            stability = replan.stability
            if stability is not None:
                if stability.hard_stability_violation:
                    st.error(
                        "Locked-prefix violation detected in this repair: "
                        + "; ".join(stability.violation_reasons)
                    )
                planned_stops = (
                    stability.preserved_poi_count
                    + len(stability.removed_pois)
                )
                shifted_text = (
                    "not measurable"
                    if stability.time_shift_count is None
                    else str(stability.time_shift_count)
                )
                st.caption(
                    f"Plan stability: {stability.preserved_poi_count} of "
                    f"{planned_stops} remaining stops preserved, "
                    f"{len(stability.removed_pois)} removed, "
                    f"{len(stability.added_pois)} added, "
                    f"{shifted_text} time-shifted, "
                    f"edit distance {stability.edit_distance_score:.2f}."
                )
        elif replan.status == "no_action":
            st.info(replan.message)
        else:
            st.error(replan.message)

    successful_replan = replan is not None and replan.status == "success"
    selected = replan.repaired_schedule if successful_replan else (initial.schedule if initial is not None else None)
    save_clicked = False
    itinerary_name = ""
    if selected is not None:
        st.markdown("#### Save this itinerary")
        itinerary_name = st.text_input(
            "Itinerary name",
            placeholder="e.g. KL Heritage Day",
            max_chars=80,
        )
        save_clicked = st.button("Save itinerary", type="primary", width="stretch")
    if save_clicked and not itinerary_name.strip():
        st.error("Enter a name before saving this itinerary.")
    if save_clicked and itinerary_name.strip():
        planning_instance = st.session_state.get("planning_instance")
        saved_mode = st.session_state.get("planning_mode", "Quick preview")
        request = planning_instance.request if planning_instance is not None else None
        request_parameters = {}
        if request is not None:
            request_parameters = {
                "trip_date": request.trip_date.isoformat(),
                "start_poi_id": request.start_poi_id,
                "start_time_min": request.start_time_min,
                "end_time_min": request.end_time_min,
                "cost_budget_rm": request.cost_budget_rm,
                "mandatory_poi_ids": list(request.mandatory_poi_ids),
                "preferred_poi_ids": list(request.preferred_poi_ids),
                "weather_mode": st.session_state.get("weather_mode", "Not recorded"),
                "weather": profile.to_dict(),
            }
        metrics = {
            "total_travel_min": selected.total_travel_min,
            "total_waiting_min": selected.total_waiting_min,
            "total_visit_min": selected.total_visit_min,
            "time_used_min": selected.time_used_min,
            "finish_min": selected.finish_min,
            "total_cost_rm": selected.total_cost_rm,
        }
        if successful_replan:
            metrics.update({
                "weather_weighted_exposure_score": replan.exposure.weather_weighted_exposure_score,
                "total_outdoor_exposure_min": replan.exposure.total_outdoor_exposure_min,
                "replanning_latency_ms": replan.replanning_latency_ms,
            })
        elif initial is not None:
            metrics.update({
                "achieved_utility": initial.ga_result.selected_solution.achieved_utility,
                "generations_executed": initial.ga_result.generations_executed,
                "convergence_generation": initial.ga_result.convergence_generation,
                "unique_route_count": initial.ga_result.unique_route_count,
            })
        history.save(
            record=ItineraryRunRecord(
                planner_id="tripmates_lite" if successful_replan else "static_utility",
                seed=1001, scenario_id=replan.state.scenario_id if successful_replan else None,
                catalogue_id=POI_FILE.name, matrix_id=MATRIX_FILE.name,
                utility_rule_version=(
                    "preferred_utility_1.25x_v1"
                    if request is not None and request.preferred_poi_ids
                    else "equal_weight_v1"
                ),
                configuration_id=("quick_preview_v1" if saved_mode == "Quick preview" else "benchmark_v1"),
                trigger_state=asdict(replan.state) if successful_replan else {},
                request_parameters=request_parameters,
                itinerary_name=itinerary_name.strip(),
                fallback_status=getattr(replan.ga_result, "selection_status", None) if successful_replan else None,
            ),
            schedule=selected,
            metrics=metrics,
        )
        st.success(f"Saved ‘{itinerary_name.strip()}’. Open Saved itineraries from the sidebar to view it.")


if __name__ == "__main__":
    main()
