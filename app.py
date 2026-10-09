"""Interactive dashboard:  streamlit run app.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from main import cell_report
from ranopt import GAConfig, ModelParams, ScenarioConfig, build_scenario, optimize
from ranopt.viz import plot_comparison, plot_summary

st.set_page_config(page_title="RAN Power Optimizer", layout="wide")
st.title("AI-driven RAN energy saving")
st.caption("Synthetic 3-sector macro network (simplified 3GPP UMa propagation). "
           "A genetic algorithm tunes TX power, electrical tilt and cell sleep mode "
           "to cut energy use while protecting coverage and user throughput.")

with st.sidebar:
    st.header("Scenario")
    seed = st.number_input("Random seed", 0, 9999, 7)
    rings = st.radio("Network size", [1, 2], format_func=lambda r: {1: "7 sites / 21 cells",
                                                                    2: "19 sites / 57 cells"}[r])
    hotspots = st.slider("Traffic hotspots", 0, 8, 4)
    st.header("QoS limits")
    cov_drop = st.slider("Max coverage drop (percentage points)", 0.0, 5.0, 1.0, 0.5)
    edge_drop = st.slider("Max cell-edge throughput drop (%)", 0, 50, 15)
    mean_drop = st.slider("Max mean throughput drop (%)", 0, 50, 20)
    st.header("Genetic algorithm")
    pop = st.slider("Population", 20, 150, 80, 10)
    gens = st.slider("Generations", 20, 250, 120, 10)
    run = st.button("Run optimisation", type="primary", width="stretch")

if run:
    scn = build_scenario(ScenarioConfig(seed=int(seed), n_rings=rings, n_hotspots=hotspots))
    ga = GAConfig(pop_size=pop, generations=gens, seed=int(seed),
                  max_coverage_drop=cov_drop / 100.0, max_edge_drop=edge_drop / 100.0,
                  max_mean_drop=mean_drop / 100.0)
    bar = st.progress(0.0, text="Optimising...")
    res = optimize(scn, ModelParams(), ga,
                   progress=lambda g, n: bar.progress(g / n, text=f"Generation {g}/{n}"))
    bar.empty()
    st.session_state["run"] = (scn, res)

if "run" not in st.session_state:
    st.info("Set the scenario in the sidebar and press **Run optimisation**.")
    st.stop()

scn, res = st.session_state["run"]
b, o = res.baseline_eval.kpis, res.best_eval.kpis
saving = 100.0 * (1.0 - o.power_w / b.power_w)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Network power", f"{o.power_w / 1e3:.2f} kW", f"-{saving:.1f} %", delta_color="inverse")
c2.metric("Cells asleep", f"{int((~res.best.active).sum())} / {scn.n_cells}")
c3.metric("Coverage", f"{o.coverage * 100:.1f} %", f"{(o.coverage - b.coverage) * 100:+.1f} pp")
c4.metric("Cell-edge throughput", f"{o.edge_tput_mbps:.2f} Mbit/s",
          f"{(o.edge_tput_mbps / b.edge_tput_mbps - 1) * 100:+.0f} %")
c5.metric("Mean throughput", f"{o.mean_tput_mbps:.2f} Mbit/s",
          f"{(o.mean_tput_mbps / b.mean_tput_mbps - 1) * 100:+.0f} %")

tab1, tab2, tab3 = st.tabs(["Coverage maps", "Optimisation summary", "Cell report"])
with tab1:
    fig = plot_comparison(scn, res.baseline, res.baseline_eval.maps, res.best, res.best_eval.maps)
    st.pyplot(fig)
    plt.close(fig)
with tab2:
    fig = plot_summary(scn, res.history, b, o, res.baseline, res.best)
    st.pyplot(fig)
    plt.close(fig)
with tab3:
    df: pd.DataFrame = cell_report(scn, res)
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "cell_report.csv", "text/csv")
