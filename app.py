"""EV Smart Routing – research web app (Streamlit). Run:  streamlit run app.py"""
import time
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from engine.graph import build_network
from engine.traffic import Snapshot
from engine.charging import make_stations, resolve_ports_up, charge_minutes, expected_wait_min
from engine.ev_problem import EVConfig, EVProblem, front_array
from engine.optimizer import METHODS, GREEDY, solve, solution_payload, benchmark
from engine.firefly import run_fa_ts

st.set_page_config(page_title="EV Smart Routing", page_icon="⚡", layout="wide")
PAGES = ["1 · Dashboard", "2 · EV Trip Configuration", "3 · Optimization Engine", "4 · Results & Route Analysis", "5 · Algorithm Performance"]
DEF = dict(src=None, dst=None, soc_pct=35, battery=30.0, cons=0.20, reserve=0.10, vmax=100.0, knee=0.8, dep="08:00", session=30.0,
           stops=3, n_st=24, st_seed=42, av_mode="random", p_down=0.10, av_seed=1, manual=None, knn=6, kmin=6.0, kmax=60.0, data_dir="data")
S = st.session_state.setdefault("cfg", dict(DEF))
COL = {"FA-TS (proposed)": "#d1495b", "FA only": "#f29e4c", "TS only": "#2a9d8f", "NSGA-II": "#3a6ea5", "Random search": "#999999",
       "SP + Nearest station": "#8e44ad", "SP + Min-detour station": "#16a085", "SP + Least-delay station": "#c0392b", "Fastest-path + Nearest": "#7f8c8d"}


# ---------------------------------------------------------------- cached builders
@st.cache_resource(show_spinner="Building road graph…")
def get_network(data_dir, knn, kmin, kmax): return build_network(data_dir, knn, kmin, kmax)


@st.cache_resource(show_spinner="Computing congestion-aware shortest paths…")
def get_snapshot(_net, key, cons, hhmm, metric): return Snapshot(_net, cons, hhmm, metric)


@st.cache_resource
def get_stations(_net, key, n, seed, hhmm):
    return make_stations(_net, get_snapshot(_net, key, S["cons"], hhmm, "distance").D, n, seed)


def num(label, key, **kw):
    v = st.number_input(label, value=type(DEF[key])(S[key]), **kw); S[key] = v; return v


net = get_network(S["data_dir"], int(S["knn"]), float(S["kmin"]), float(S["kmax"]))
NK = (S["data_dir"], int(S["knn"]), float(S["kmin"]), float(S["kmax"]))
stations = get_stations(net, NK, int(S["n_st"]), int(S["st_seed"]), "08:00")
cfg = EVConfig(S["battery"], S["cons"], S["reserve"], S["vmax"], S["knee"], S["session"], int(S["stops"]))
snap = get_snapshot(net, NK, S["cons"], S["dep"], "distance")
stn = set(stations["node"].tolist())
if S["src"] is None or S["dst"] is None:
    D0 = snap.D.copy(); ok = [i for i in range(net.N) if i not in stn]; sub = D0[np.ix_(ok, ok)]
    a, b = np.unravel_index(np.argmax(sub), sub.shape); S["src"], S["dst"] = int(ok[a]), int(ok[b])
ports_up = resolve_ports_up(stations, S["av_mode"], S["p_down"], S["av_seed"], S["manual"] if S["manual"] and len(S["manual"]) == len(stations["node"]) else None)
soc0 = S["soc_pct"] / 100 * cfg.cap


def build_problem(hhmm=None, metric="distance"):
    sn = snap if (hhmm is None and metric == "distance") else get_snapshot(net, NK, S["cons"], hhmm or S["dep"], metric)
    return EVProblem(sn, stations, ports_up, S["src"], S["dst"], soc0, cfg)


# ---------------------------------------------------------------- figure helpers
def network_fig(sn, paths=(), stops=(), height=520, congestion=True, title=None):
    P, fig = net.pos, go.Figure()
    ratio = np.array([0.5 * (sn.v_node[u] + sn.v_node[v]) / (0.5 * (net.v_free[u] + net.v_free[v])) for u, v in net.edges])
    bins = [(0.0, 0.6, "#d1495b", "heavy congestion"), (0.6, 0.8, "#f2c94c", "slow"), (0.8, 9, "#9bb7a0", "free flow")] if congestion else [(0, 9, "#bbbbbb", "road")]
    for lo, hi, c, nm in bins:
        xs, ys = [], []
        for (u, v), r in zip(net.edges, ratio):
            if lo <= r < hi: xs += [P[u, 0], P[v, 0], None]; ys += [P[u, 1], P[v, 1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=c, width=1.5), name=nm, hoverinfo="skip"))
    for i, (pth, col, nm, dash) in enumerate(paths):
        fig.add_trace(go.Scatter(x=P[pth, 0], y=P[pth, 1], mode="lines", line=dict(color=col, width=4 if dash is None else 2, dash=dash), name=nm))
    sn_ = stations["node"]
    fig.add_trace(go.Scatter(x=P[sn_, 0], y=P[sn_, 1], mode="markers", marker=dict(size=9, color="gold", line=dict(color="black", width=1)), name="charging station",
                             text=[f"Station {k} · {stations['power'][k]:.0f} kW · {ports_up[k]}/{stations['ports'][k]} ports up" for k in range(len(sn_))], hoverinfo="text"))
    for nd, lab, col in stops: fig.add_trace(go.Scatter(x=[P[nd, 0]], y=[P[nd, 1]], mode="markers+text", text=[lab], textposition="top right", marker=dict(size=15, color=col, symbol="star" if lab == "Dest" else "circle"), showlegend=False))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30 if title else 0, b=0), title=title, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), legend=dict(orientation="h"))
    return fig


def front_fig(items, selected=None, height=420):
    fig = go.Figure()
    for nm, res in items.items():
        F = front_array(res["front"])
        if not len(F): continue
        single = nm in GREEDY or nm.startswith("Fastest")
        fig.add_trace(go.Scatter(x=F[:, 0], y=F[:, 1], name=nm, mode="markers" if single else "lines+markers", line=dict(shape="hv", color=COL.get(nm), width=3 if "proposed" in nm else 1.5),
                                 marker=dict(size=11 if single else 6, symbol="triangle-up" if single else "circle", color=COL.get(nm))))
    if selected is not None: fig.add_trace(go.Scatter(x=[selected[0]], y=[selected[1]], mode="markers", marker=dict(size=18, symbol="star", color="black"), name="selected"))
    fig.update_layout(height=height, xaxis_title="total distance (km)", yaxis_title="expected charging delay (min)", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚡ EV Smart Routing")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    (st.warning if net.source == "synthetic" else st.success)(f"Data: **{net.source}**" + (" – put metr-la.h5 & adj.npz in data/ for real results" if net.source == "synthetic" else ""))
    st.caption(f"{net.N} nodes · {len(net.edges)} edges · {len(stations['node'])} stations · {'geographic' if net.geo else 'schematic'} map")
    with st.expander("Graph settings"):
        S["knn"] = st.number_input("kNN neighbours", 3, 20, int(S["knn"])); S["kmin"] = st.number_input("Edge min (km)", 0.5, 100.0, float(S["kmin"]))
        S["kmax"] = st.number_input("Edge max (km)", 1.0, 200.0, float(S["kmax"])); S["data_dir"] = st.text_input("Data folder", S["data_dir"])
        st.caption("Changing these rebuilds the network – re-select origin/destination.")

# ================================================================ 1 DASHBOARD
if page == PAGES[0]:
    st.header("Dashboard")
    run = st.session_state.get("run")
    c = st.columns(6)
    for col, (l, v) in zip(c, [("Road nodes", net.N), ("Road segments", len(net.edges)), ("Charging stations", len(stations["node"])), (f"Mean speed {S['dep']}", f"{snap.v_node.mean():.0f} km/h"),
                               ("Mean congestion idx", f"{snap.cong.mean():.2f}"), ("Ports in service", f"{ports_up.sum()}/{stations['ports'].sum()}")]): col.metric(l, v)
    a, b = st.columns([3, 2])
    with a: st.plotly_chart(network_fig(snap, stops=[(S["src"], "Origin", "#2a9d8f"), (S["dst"], "Dest", "#8e44ad")], title=f"Traffic conditions at {S['dep']}"), width="stretch")
    with b:
        hrs = np.arange(288) / 12; pr = net.profile_kmh; f = go.Figure()
        f.add_trace(go.Scatter(x=hrs, y=np.percentile(pr, 90, axis=1), line=dict(width=0), showlegend=False)); f.add_trace(go.Scatter(x=hrs, y=np.percentile(pr, 10, axis=1), fill="tonexty", line=dict(width=0), name="10–90 % sensors"))
        f.add_trace(go.Scatter(x=hrs, y=pr.mean(1), name="mean speed", line=dict(width=3))); hh = int(S["dep"][:2]) + int(S["dep"][3:5]) / 60; f.add_vline(x=hh, line_dash="dash", line_color="#d1495b", annotation_text="departure")
        f.update_layout(height=250, margin=dict(l=0, r=0, t=25, b=0), title="Typical weekday speed (km/h)"); st.plotly_chart(f, width="stretch")
        w, _ = __import__("engine.charging", fromlist=["station_waits"]).station_waits(stations, ports_up, snap, S["session"])
        f = go.Figure(go.Bar(x=list(range(len(w))), y=w, marker_color="#3a6ea5")); f.update_layout(height=220, margin=dict(l=0, r=0, t=25, b=0), title="Expected queue wait per station (min, Erlang-C)", xaxis_title="station"); st.plotly_chart(f, width="stretch")
    st.subheader("System performance (last optimization run)")
    if run:
        st.dataframe(pd.DataFrame([dict(method=m, front_size=len(r["front"]), best_dist_km=min(x[0] for x in r["front"]) if r["front"] else None, best_delay_min=min(x[1] for x in r["front"]) if r["front"] else None, evals=r["evals"], time_s=round(r["time"], 2)) for m, r in run["results"].items()]), hide_index=True, width="stretch")
    else: st.info("No optimization yet – configure a trip (page 2) and run the engine (page 3).")

# ================================================================ 2 TRIP CONFIG
elif page == PAGES[1]:
    st.header("EV Trip Configuration")
    lab = lambda i: f"Node {i} · sensor {net.sensor_ids[i]}" + (" (station)" if i in stn else "")
    a, b, c = st.columns(3)
    with a:
        st.subheader("Route")
        S["src"] = st.selectbox("Origin", range(net.N), index=int(S["src"]), format_func=lab); S["dst"] = st.selectbox("Destination", range(net.N), index=int(S["dst"]), format_func=lab)
        t = st.time_input("Departure time", value=pd.to_datetime(S["dep"]).time(), step=300); S["dep"] = t.strftime("%H:%M")
    with b:
        st.subheader("Vehicle")
        S["soc_pct"] = st.slider("Initial battery SoC (%)", 5, 100, int(S["soc_pct"])); num("Battery capacity (kWh)", "battery", min_value=10.0, max_value=200.0, step=1.0)
        num("Consumption (kWh/km)", "cons", min_value=0.05, max_value=0.6, step=0.01); num("Reserve fraction", "reserve", min_value=0.0, max_value=0.5, step=0.01)
        num("Max DC charge power (kW)", "vmax", min_value=20.0, max_value=350.0, step=10.0); num("Taper knee (SoC)", "knee", min_value=0.5, max_value=1.0, step=0.05); S["stops"] = st.number_input("Max charging stops", 0, 5, int(S["stops"]))
    with c:
        st.subheader("Stations & availability")
        S["n_st"] = st.number_input("Number of stations", 4, 60, int(S["n_st"])); S["st_seed"] = st.number_input("Station layout seed", 0, 9999, int(S["st_seed"])); num("Charging session (min)", "session", min_value=10.0, max_value=120.0, step=5.0)
        S["av_mode"] = st.radio("Availability", ["all_up", "random", "manual"], index=["all_up", "random", "manual"].index(S["av_mode"]), horizontal=True, format_func={"all_up": "All ports up", "random": "Random failures", "manual": "Manual"}.get)
        if S["av_mode"] == "random": S["p_down"] = st.slider("Port-down probability", 0.0, 0.6, float(S["p_down"]), 0.05); S["av_seed"] = st.number_input("Availability seed", 0, 9999, int(S["av_seed"]))
    st.info("Settings are applied on the next interaction; the summary below reflects the current values.")
    if S["av_mode"] == "manual":
        df = pd.DataFrame(dict(station=range(len(stations["node"])), node=stations["node"], power_kw=stations["power"], ports=stations["ports"], ports_in_service=(S["manual"] if S["manual"] and len(S["manual"]) == len(stations["node"]) else stations["ports"].tolist())))
        ed = st.data_editor(df, disabled=["station", "node", "power_kw", "ports"], hide_index=True, width="stretch"); S["manual"] = ed["ports_in_service"].clip(0, ed["ports"]).tolist()
    d_km, d_kwh = snap.D[S["src"], S["dst"]], snap.E[S["src"], S["dst"]]
    m = st.columns(4); m[0].metric("Direct distance", f"{d_km:.0f} km"); m[1].metric("Direct energy need", f"{d_kwh:.1f} kWh"); m[2].metric("Usable battery now", f"{soc0 - cfg.reserve:.1f} kWh")
    m[3].metric("Charging required?", "Yes" if soc0 - d_kwh < cfg.reserve else "No")
    if S["src"] == S["dst"]: st.error("Origin and destination must differ.")
    st.plotly_chart(network_fig(snap, paths=[(snap.path(S["src"], S["dst"]), "#3a6ea5", "direct shortest path", "dash")] if S["src"] != S["dst"] else [], stops=[(S["src"], "Origin", "#2a9d8f"), (S["dst"], "Dest", "#8e44ad")], height=450), width="stretch")

# ================================================================ 3 OPTIMIZATION ENGINE
elif page == PAGES[2]:
    st.header("Optimization Engine")
    st.caption(f"Problem: node {S['src']} → node {S['dst']}, SoC {S['soc_pct']} % ({soc0:.1f} kWh), battery {cfg.cap:.0f} kWh, departure {S['dep']}. Only this problem is solved.")
    meths = st.multiselect("Algorithms to run", list(METHODS) + list(GREEDY), default=["FA-TS (proposed)"])
    a, b, c = st.columns(3)
    budget = a.slider("Optimization budget (objective evaluations)", 500, 20000, 3000, 250); seed = b.number_input("Random seed", 0, 9999, 0)
    with c.expander("FA-TS parameters"): ts = st.slider("Tabu share of budget", 0.0, 0.9, 0.6, 0.1); pop = st.slider("Firefly population", 8, 60, 24, 2)
    if st.button("▶ Run optimization", type="primary", disabled=S["src"] == S["dst"] or not meths):
        P = build_problem(); results, payloads = {}, {}; bar = st.progress(0.0)
        for i, m in enumerate(meths):
            bar.progress(i / len(meths), f"Running {m}…")
            res = run_fa_ts(P, budget, np.random.default_rng(seed), pop=pop, ts_frac=ts) if m == "FA-TS (proposed)" else solve(P, m, budget, seed)
            results[m] = res; payloads[m] = solution_payload(P, res)
        bar.progress(1.0, "Done"); st.session_state["run"] = dict(results=results, payloads=payloads, problem=P, budget=budget, seed=seed, cfg=dict(S))
        st.success("Finished – open **4 · Results & Route Analysis**.")
    run = st.session_state.get("run")
    if run:
        st.subheader("Latest run"); st.plotly_chart(front_fig(run["results"]), width="stretch")
        st.dataframe(pd.DataFrame([dict(method=m, solutions=len(r["front"]), best_dist_km=min((x[0] for x in r["front"]), default=None), best_delay_min=min((x[1] for x in r["front"]), default=None), evals=r["evals"], time_s=round(r["time"], 2), reached_destination=len(r["front"]) > 0) for m, r in run["results"].items()]), hide_index=True, width="stretch")
        if any(len(r["front"]) == 0 for r in run["results"].values()): st.warning("A method without solutions could not reach the destination within the allowed stops/battery.")

# ================================================================ 4 RESULTS
elif page == PAGES[3]:
    st.header("Results and Route Analysis"); run = st.session_state.get("run")
    if not run: st.info("Run an optimization first (page 3).")
    else:
        opts = [m for m in run["results"] if run["results"][m]["front"]]
        if not opts: st.error("No feasible route found. Increase initial SoC / battery, allow more stops, or add stations.")
        else:
            m = st.selectbox("Method", opts, index=opts.index("FA-TS (proposed)") if "FA-TS (proposed)" in opts else 0); pl = run["payloads"][m]; P = run["problem"]; n = len(pl["solutions"])
            c1, c2 = st.columns([3, 1]); idx = c1.slider("Pareto solution (left = shortest, right = least charging delay)", 0, max(n - 1, 0), pl["knee"] if n > 1 else 0) if n > 1 else 0
            c2.write(""); c2.caption(f"Recommended compromise (knee): solution {pl['knee']}")
            d = pl["details"][idx]; sol = pl["solutions"][idx]

            k = st.columns(6)
            metrics = [("Distance", f"{d['total_km']:.1f} km"), ("Energy", f"{d['energy_kwh']:.1f} kWh"), ("Stops", len(d["stops"])), ("Waiting", f"{d['wait_min']:.1f} min"), ("Charging", f"{d['charge_min']:.1f} min"), ("Total time", f"{d['total_min']:.0f} min")]
            for x, (l, v) in zip(k, metrics):
                x.metric(l, v)


            
            sn = P.snap; paths = [(sn.path(a, b), "#d1495b", "optimised route", None) for a, b in zip(d["waypoints"][:-1], d["waypoints"][1:])]
            paths = [(p, c, nm if i == 0 else None, dash) for i, (p, c, nm, dash) in enumerate(paths)]; paths = [(p, c, nm or "", dash) for p, c, nm, dash in paths]
            marks = [(P.src, "Origin", "#2a9d8f"), (P.dst, "Dest", "#8e44ad")] + [(s["node"], f"S{i + 1} → {P.LEVELS[sol[i][1]] * 100:.0f}%", "#d1495b") for i, s in enumerate(d["stops"])]
            a, b = st.columns(2)
            with a:
                st.plotly_chart(front_fig({m: run["results"][m]}, selected=(pl["front"]["distance_km"][idx], pl["front"]["delay_min"][idx]), height=360), width="stretch")
                st.plotly_chart(network_fig(sn, paths + [(sn.path(P.src, P.dst), "#3a6ea5", "direct (infeasible)", "dash")], marks, height=420), width="stretch")
            with b:
                f = go.Figure(go.Scatter(x=d["soc_x"], y=d["soc_y"], mode="lines+markers", name="SoC", line=dict(color="#2a9d8f", width=3)))
                f.add_hline(y=cfg.reserve, line_dash="dash", line_color="#d1495b", annotation_text="reserve"); f.add_hline(y=cfg.cap, line_dash="dot", annotation_text="capacity")
                f.update_layout(height=300, title="State of charge along route (vertical jumps = charging)", xaxis_title="distance (km)", yaxis_title="kWh", margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(f, width="stretch")
                f = go.Figure([go.Bar(x=[v], y=["time"], orientation="h", name=l, marker_color=c) for l, v, c in [("driving", d["drive_min"], "#3a6ea5"), ("waiting", d["wait_min"], "#f29e4c"), ("charging", d["charge_min"], "#2a9d8f")]]); f.update_layout(barmode="stack", height=130, title="Trip time breakdown (min)", margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(f, width="stretch")
                if d["stops"]: st.dataframe(pd.DataFrame([dict(stop=f"S{i + 1}", station=s["station"], node=s["node"], power_kw=s["power_kw"], ports_up=f"{s['ports_up']}/{s['ports_total']}", arrive_soc_kwh=round(s["arrive_soc"], 1), charge_to_kwh=round(s["charge_to"], 1), wait_min=round(s["wait_min"], 1), charge_min=round(s["charge_min"], 1)) for i, s in enumerate(d["stops"])]), hide_index=True, width="stretch")
                else: st.success("No charging stop needed for this solution.")
            st.subheader("Reactive baselines on the same problem")
            rows = []
            from engine.ev_problem import greedy_baseline
            for nm, md in GREEDY.items():
                g = greedy_baseline(P, md)
                if g is None: rows.append(dict(strategy=nm, distance_km=None, delay_min=None, note="cannot reach destination")); continue
                f1, f2, _, _ = P.eval_raw(g); dom = any(x <= f1 + 1e-9 and y <= f2 + 1e-9 for x, y in zip(pl["front"]["distance_km"], pl["front"]["delay_min"]))
                rows.append(dict(strategy=nm, distance_km=round(f1, 1), delay_min=round(f2, 1), note="dominated by the optimised front" if dom else "not dominated"))
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ================================================================ 5 ALGORITHM PERFORMANCE
else:
    st.header("Algorithm Performance"); st.caption("Equal evaluation budget on the currently selected problem (page 2). Quality = hypervolume ratio vs the reference front.")
    allm = list(METHODS) + list(GREEDY) + ["Fastest-path + Nearest"]
    meths = st.multiselect("Methods", allm, default=list(METHODS) + ["SP + Nearest station"])
    a, b, c, d = st.columns(4)
    budget = a.slider("Budget per run", 500, 10000, 2000, 250); ns = b.slider("Seeds per stochastic method", 1, 15, 4); seed = c.number_input("Base seed", 0, 9999, 0)
    ex = d.checkbox("Exact reference (enumerate all)", value=len(stations["node"]) <= 12, help="Exact front by exhaustive search; slow for many stations. Otherwise the pooled non-dominated set is the reference.")
    if st.button("▶ Run benchmark", type="primary", disabled=S["src"] == S["dst"]):
        bar = st.progress(0.0); P = build_problem(); Pf = build_problem(metric="time") if "Fastest-path + Nearest" in meths else None
        st.session_state["bench"] = benchmark(P, meths, budget, ns, int(seed), exact=ex, P_fast=Pf, progress=lambda f, t: bar.progress(min(f, 1.0), t)); bar.progress(1.0, "Done")
    B = st.session_state.get("bench")
    if B:
        df = pd.DataFrame(B["records"]); st.caption(("Exact" if B["exact"] else "Pooled non-dominated") + f" reference front: {len(B['ref_front'])} points")
        g = df.groupby("method").agg(HV_ratio=("hv_ratio", "mean"), HV_sd=("hv_ratio", "std"), front_size=("front_size", "mean"), best_dist_km=("best_dist_km", "mean"), best_delay_min=("best_delay_min", "mean"), time_s=("time_s", "mean"), solved=("solved", "mean")).round(3).sort_values("HV_ratio", ascending=False)
        st.dataframe(g, width="stretch")
        a, b = st.columns(2)
        with a:
            f = go.Figure([go.Box(y=df[df.method == m].hv_ratio, name=m, marker_color=COL.get(m), boxmean=True) for m in g.index]); f.update_layout(height=380, title="Pareto-front quality (HV ratio)", showlegend=False, margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(f, width="stretch")
            f = go.Figure(go.Bar(x=g.index, y=g.time_s, marker_color=[COL.get(m) for m in g.index])); f.update_layout(height=300, title="Mean runtime (s)", margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(f, width="stretch")
        with b:
            f = go.Figure()
            for m, cs in B["curves"].items():
                if not cs: continue
                n = min(len(c) for c in cs); f.add_trace(go.Scatter(x=[p[0] for p in cs[0][:n]], y=np.mean([[p[1] for p in c[:n]] for c in cs], 0), name=m, line=dict(color=COL.get(m), width=4 if "proposed" in m else 2)))
            f.update_layout(height=380, title="Convergence (mean HV ratio vs evaluations)", xaxis_title="objective evaluations", margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(f, width="stretch")
            items = {m: rs[0] for m, rs in B["runs"].items()}
            if len(B["ref_front"]): items = {"Reference front": dict(front=[(x, y, None) for x, y in B["ref_front"]]), **items}; COL["Reference front"] = "#222222"
            st.plotly_chart(front_fig(items, height=300), width="stretch")
        if "FA-TS (proposed)" in g.index and len(g) > 1: st.info(f"FA-TS mean HV ratio {g.loc['FA-TS (proposed)', 'HV_ratio']:.3f}; best method: **{g.index[0]}**. Use several scenarios and ≥ 8 seeds (see the notebook) before drawing conclusions.")
