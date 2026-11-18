# EV Smart Routing – FA-TS research application

Web app (Streamlit + Python, runs locally) for congestion- and charging-aware EV routing with the hybrid
Firefly–Tabu Search (FA-TS) from the research notebook. The FA-TS algorithm is unchanged; the notebook
globals were converted into reusable modules.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501.

## Data
Put the Kaggle `traffic-metr-la` files in `data/`: `metr-la.h5`, `adj.npz`
(optional: `graph_sensor_locations.csv` to draw a real lon/lat map).
Without them the app uses a **synthetic fallback network** (labelled in the sidebar) – report results only from real data.

## Pages
1. Dashboard – network, congestion at the chosen departure time, stations, last-run performance
2. EV Trip Configuration – origin, destination, SoC, vehicle, departure time, station availability
3. Optimization Engine – run FA-TS (or any method) for the selected problem only
4. Results & Route Analysis – Pareto front, recommended route, stops, waiting, energy
5. Algorithm Performance – FA-TS vs NSGA-II, Random, FA-only, TS-only (HV ratio, convergence, runtime)

## Notebook → app changes
| Notebook | App |
|---|---|
| random scenarios | user-selected origin/destination |
| random availability | `all_up` / random(p_down) / manual ports-in-service table |
| fixed battery | battery, consumption, reserve, charging power from UI |
| fixed peak/off-peak | any departure time (HH:MM, 5-min resolution) |
| full experiment | only the selected problem is solved |
| matplotlib | Plotly figures built from plain-dict results |
| globals | `engine/` modules (`EVConfig`, `EVProblem(snap, …, cfg)`) |

## Notes
* Sensor nodes have no coordinates in `adj.npz`; the map is a schematic graph unless `graph_sensor_locations.csv` is supplied.
* Stations/EVs are an overlay on METR-LA (assumption). Edge lengths are a rescaled kernel inversion (6–60 km by default).
* The "exact" reference enumerates all solutions: fine for ≲ 24 stations and 3 stops, slow beyond that.
