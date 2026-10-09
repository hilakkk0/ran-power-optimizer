from ranopt import (GAConfig, ModelParams, ScenarioConfig, baseline_state,
                    build_scenario, evaluate, optimize)


def _scn():
    return build_scenario(ScenarioConfig(seed=1))


def test_baseline_values_are_physically_plausible():
    scn, p = _scn(), ModelParams()
    ev = evaluate(scn, p, baseline_state(scn, p), maps=True)
    assert -140 < ev.maps["rsrp"].min() and ev.maps["rsrp"].max() < -40
    assert ev.maps["sinr"].max() < 40
    assert 0.8 < ev.kpis.coverage <= 1.0
    assert abs(ev.cell_users.sum() - scn.traffic[ev.maps["covered"].ravel()].sum()) < 1e-6


def test_lower_power_never_increases_power_consumption():
    scn, p = _scn(), ModelParams()
    st = baseline_state(scn, p)
    high = evaluate(scn, p, st).kpis.power_w
    st.power_dbm -= 6
    assert evaluate(scn, p, st).kpis.power_w < high


def test_sleeping_cell_saves_energy_and_serves_nobody():
    scn, p = _scn(), ModelParams()
    st = baseline_state(scn, p)
    full = evaluate(scn, p, st).kpis
    st.active[0] = False
    ev = evaluate(scn, p, st)
    assert ev.kpis.power_w < full.power_w
    assert ev.cell_users[0] == 0.0


def test_optimizer_is_never_worse_than_baseline():
    scn = _scn()
    res = optimize(scn, ModelParams(), GAConfig(pop_size=20, generations=8, seed=3))
    assert res.history["best_fitness"][-1] <= 1.0 + 1e-9
    assert res.best_eval.kpis.power_w <= res.baseline_eval.kpis.power_w + 1e-6
