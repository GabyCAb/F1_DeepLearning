"""
Previsão de Top 10 para as próximas corridas da temporada, usando o modelo salvo
(f1_top10_model.keras) — sem retreinar.

Reproduz exatamente o pipeline do notebook F1_Classificacao_Top10.ipynb:
  - mesmas features e mesmo pré-processamento (StandardScaler + OneHotEncoder,
    ajustados nas temporadas 1950–2023);
  - "forma" do piloto/equipe = médias da temporada até a última corrida disputada;
  - grid ainda desconhecido -> substituído pela média de grid do piloto na temporada.

Diferenças em relação à versão do notebook:
  - a última rodada disputada e as próximas rodadas são detectadas automaticamente
    (não ficam fixas em 12 / 13-15);
  - a escalação de cada equipe é a dos 2 pilotos com mais largadas por ela nas
    últimas 5 rodadas (evita contar substitutos de uma corrida só) e aceita
    trocas confirmadas em SUBSTITUICOES;
  - também gera uma variante "forma recente" (grid médio das últimas 5 corridas).

Uso (na raiz do projeto):  python prever_proximas_corridas.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorflow import keras  # noqa: E402

DATA_DIR = "dataframes"
SEASON = 2026
N_PROXIMAS = 4          # quantas corridas futuras prever
JANELA_RECENTE = 5      # rodadas usadas para escalação e "forma recente"

# Trocas de piloto já confirmadas para as próximas corridas:
# piloto_que_sai -> (piloto_que_entra, nacionalidade).
# Haas: Ocon fora pelo resto de 2026; Rafael Câmara estreia no GP do Brasil
# (confirmado só para o Brasil até 03/11/2026 — mantido nas demais por hipótese).
SUBSTITUICOES = {
    "ocon": ("rafael_camara", "Brazilian"),
    "ryo_hirakawa": ("rafael_camara", "Brazilian"),
}

NUMERIC = ["grid", "lat", "long", "season", "driver_avg_pos_season", "constructor_avg_points_season"]
CATEGORICAL = ["constructor_id", "circuit_id", "driver_nationality"]


def carregar():
    results = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    races = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
    df = results.merge(races[["season", "round", "circuit_id", "country", "lat", "long"]],
                       on=["season", "round"], how="left")
    df["target_top10"] = (df["position"] <= 10).astype(int)
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["driver_avg_pos_season"] = (df.groupby(["season", "driver_id"])["position"]
                                   .apply(lambda s: s.shift().expanding().mean()).reset_index(drop=True))
    df["constructor_avg_points_season"] = (df.groupby(["season", "constructor_id"])["points"]
                                           .apply(lambda s: s.shift().expanding().mean()).reset_index(drop=True))
    df["driver_avg_pos_season"] = df["driver_avg_pos_season"].fillna(df["position"].median())
    df["constructor_avg_points_season"] = df["constructor_avg_points_season"].fillna(df["points"].median())
    return results, races, df


def main():
    results, races, df = carregar()
    model_df = df[NUMERIC + CATEGORICAL + ["target_top10", "round"]].dropna(subset=NUMERIC)
    train_df = model_df[model_df["season"] <= 2023]

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    pre.fit(train_df[NUMERIC + CATEGORICAL])
    model = keras.models.load_model("f1_top10_model.keras")

    s = df[df.season == SEASON]
    last_round = int(s["round"].max())
    last_race = s[s["round"] == last_round]["race_name"].iloc[0]
    print(f"Última corrida disputada: rodada {last_round} — {last_race}")

    # ---------- avaliação do modelo nas corridas de 2026 (com grid real) ----------
    s_eval = model_df[model_df.season == SEASON]
    p_eval = model.predict(pre.transform(s_eval[NUMERIC + CATEGORICAL]), verbose=0).ravel()
    s_eval = s_eval.assign(prob=p_eval)

    # ---------- estado atual da temporada ----------
    recent = s[s["round"] > last_round - JANELA_RECENTE]
    driver_avg_pos = s.groupby("driver_id")["position"].mean()
    driver_avg_grid = s.groupby("driver_id")["grid"].mean()
    driver_recent_grid = recent.groupby("driver_id")["grid"].mean()
    constructor_avg_pts = s.groupby("constructor_id")["points"].mean()
    driver_nat = s.groupby("driver_id")["driver_nationality"].last()
    constructor_avg_grid = s.groupby("constructor_id")["grid"].mean()
    constructor_avg_pos = s.groupby("constructor_id")["position"].mean()
    constructor_recent_grid = recent.groupby("constructor_id")["grid"].mean()

    # escalação: 2 pilotos com mais largadas por equipe nas últimas rodadas
    starts = (recent.groupby(["constructor_id", "driver_id"]).size()
              .rename("n").reset_index().sort_values(["constructor_id", "n"], ascending=[True, False]))
    lineup = []
    for cid, g in starts.groupby("constructor_id"):
        for drv in g["driver_id"].head(2):
            novo, nat = SUBSTITUICOES.get(drv, (drv, None))
            lineup.append((novo, cid, nat))
    lineup = list(dict.fromkeys(lineup))

    upcoming = races[(races.season == SEASON) & (races["round"] > last_round)].sort_values("round").head(N_PROXIMAS)

    preds = []
    for _, race in upcoming.iterrows():
        rows = []
        for drv, cid, nat in lineup:
            estreante = drv not in driver_avg_pos.index
            rows.append({
                "driver_id": drv,
                # estreante sem histórico na temporada: usa a média do carro (grid e posição)
                "grid": constructor_avg_grid[cid] if estreante else driver_avg_grid[drv],
                "grid_recente": constructor_recent_grid[cid] if estreante else driver_recent_grid.get(drv, driver_avg_grid[drv]),
                "lat": race["lat"], "long": race["long"], "season": SEASON,
                "driver_avg_pos_season": constructor_avg_pos[cid] if estreante else driver_avg_pos[drv],
                "constructor_avg_points_season": constructor_avg_pts[cid],
                "constructor_id": cid, "circuit_id": race["circuit_id"],
                "driver_nationality": nat or driver_nat[drv],
            })
        X = pd.DataFrame(rows)
        X["prob_top10"] = model.predict(pre.transform(X[NUMERIC + CATEGORICAL]), verbose=0).ravel()
        Xr = X.assign(grid=X["grid_recente"])
        X["prob_top10_forma_recente"] = model.predict(pre.transform(Xr[NUMERIC + CATEGORICAL]), verbose=0).ravel()
        X["race_name"], X["round"], X["date"] = race["race_name"], int(race["round"]), race["date"]
        preds.append(X)
    result = pd.concat(preds, ignore_index=True)

    for rnd, sub in result.groupby("round"):
        sub = sub.sort_values("prob_top10", ascending=False)
        print(f"\n=== {sub['race_name'].iloc[0]} (rodada {rnd}, {sub['date'].iloc[0]}) ===")
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            mark = "  <- Top10 previsto" if i <= 10 else ""
            print(f"{i:2d}. {r.driver_id:<16s} {r.constructor_id:<13s} {r.prob_top10:6.1%}  "
                  f"(forma recente {r.prob_top10_forma_recente:6.1%}){mark}")

    result.to_csv("predicoes_proximas_corridas_2026.csv", index=False)

    # ---------- dashboard_data.json + dashboard.html ----------
    with open("dashboard_data.json", encoding="utf-8") as f:
        data = json.load(f)

    def nome(d):
        return {"rafael_camara": "Rafael Câmara", "max_verstappen": "Max Verstappen",
                "arvid_lindblad": "Arvid Lindblad", "hulkenberg": "Hülkenberg",
                "perez": "Pérez"}.get(d, d.replace("_", " ").title())

    def equipe(c):
        return {"mclaren": "McLaren", "rb": "Racing Bulls", "red_bull": "Red Bull",
                "aston_martin": "Aston Martin"}.get(c, c.replace("_", " ").title())

    data["predictions"] = [{
        "round": int(rnd), "race_name": sub["race_name"].iloc[0], "date": str(sub["date"].iloc[0]),
        "drivers": [{"driver": nome(r.driver_id), "constructor": equipe(r.constructor_id),
                     "prob": round(float(r.prob_top10), 4)}
                    for _, r in sub.sort_values("prob_top10", ascending=False).head(10).iterrows()],
    } for rnd, sub in result.groupby("round")]
    data["data_status"] = {"last_round": last_round, "last_race": last_race,
                           "season_eval_auc_2026": None}
    from sklearn.metrics import accuracy_score, roc_auc_score
    data["data_status"]["season_eval_auc_2026"] = round(float(roc_auc_score(s_eval.target_top10, s_eval.prob)), 4)
    data["data_status"]["season_eval_acc_2026"] = round(float(accuracy_score(s_eval.target_top10, s_eval.prob >= .5)), 4)
    with open("dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if os.path.exists("dashboard.html"):
        html = open("dashboard.html", encoding="utf-8").read()
        html = re.sub(r"const DATA = \{.*?\};\n", lambda m: "const DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
                      html, count=1, flags=re.S)
        open("dashboard.html", "w", encoding="utf-8").write(html)

    print(f"\n2026 até a rodada {last_round} (grid real): acurácia {data['data_status']['season_eval_acc_2026']:.1%}, "
          f"AUC {data['data_status']['season_eval_auc_2026']:.3f}")
    print("Exportado: predicoes_proximas_corridas_2026.csv, dashboard_data.json, dashboard.html")
    return s_eval, result


if __name__ == "__main__":
    main()
