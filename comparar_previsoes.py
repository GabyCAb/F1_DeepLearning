"""
Previsões x resultados reais — temporada 2026.

Para cada corrida já disputada, refaz a previsão de Top 10 com o modelo salvo
(f1_top10_model.keras) usando só o que se sabia ANTES da corrida, e compara com o
resultado real. Dois cenários:

  - "Com grid real":  grid oficial de largada + forma da temporada até a corrida anterior
                      (é o que o modelo vê no teste — previsão feita depois do classificatório).
  - "Antes do classificatório": grid substituído pela média de grid do piloto nas corridas
                      anteriores (a mesma aproximação usada nas previsões de corridas futuras).

O 2026 inteiro está fora do treino (treino <= 2023), então é um teste honesto do modelo.
Gera comparacao_previsoes.json e atualiza os dados embutidos em comparacao_previsoes.html.

Uso (na raiz do projeto, depois de atualizar os dados):  python comparar_previsoes.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import prever_proximas_corridas as P

SEASON = P.SEASON
HTML = "comparacao_previsoes.html"


def prior_mean(df, keys, col):
    return df.groupby(keys)[col].transform(lambda s: s.shift().expanding().mean())


def main():
    _, races, df = P.carregar()
    feats = P.NUMERIC + P.CATEGORICAL
    model_df = df.dropna(subset=P.NUMERIC)
    pre = ColumnTransformer([
        ("num", StandardScaler(), P.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), P.CATEGORICAL),
    ]).fit(model_df[model_df.season <= 2023][feats])
    model = P.keras.models.load_model("f1_top10_model.keras")

    s = df[df.season == SEASON].copy()
    # cenário "antes do classificatório": grid = média de grid do piloto nas corridas anteriores;
    # com menos de MIN_CORRIDAS corridas anteriores, usa as médias do carro (igual ao script de previsão)
    s["prior_starts"] = s.groupby("driver_id").cumcount()
    s["drv_prior_grid"] = prior_mean(s, ["driver_id"], "grid")
    s["car_prior_grid"] = prior_mean(s, ["constructor_id"], "grid")
    s["car_prior_pos"] = prior_mean(s, ["constructor_id"], "position")
    poucas = s["prior_starts"] < P.MIN_CORRIDAS
    s["grid_pre"] = np.where(poucas, s["car_prior_grid"], s["drv_prior_grid"])
    s["avgpos_pre"] = np.where(poucas, s["car_prior_pos"], s["driver_avg_pos_season"])

    s["p_real"] = model.predict(pre.transform(s[feats]), verbose=0).ravel()
    tem_pre = s["grid_pre"].notna()
    Xpre = s.loc[tem_pre].assign(grid=s.loc[tem_pre, "grid_pre"], driver_avg_pos_season=s.loc[tem_pre, "avgpos_pre"])
    s["p_pre"] = np.nan
    s.loc[tem_pre, "p_pre"] = model.predict(pre.transform(Xpre[feats]), verbose=0).ravel()

    def nome(d):
        return {"rafael_camara": "Rafael Câmara", "max_verstappen": "Max Verstappen",
                "arvid_lindblad": "Arvid Lindblad", "hulkenberg": "Hülkenberg", "perez": "Pérez",
                "ryo_hirakawa": "Hirakawa"}.get(d, d.replace("_", " ").title())

    def equipe(c):
        return {"mclaren": "McLaren", "rb": "Racing Bulls", "red_bull": "Red Bull",
                "aston_martin": "Aston Martin"}.get(c, c.replace("_", " ").title())

    def resultado(r):
        if pd.notna(r.position) and r.position_text.isdigit():
            return f"P{int(r.position)}"
        return {"R": "Abandono", "W": "Não largou", "D": "Desclassif."}.get(r.position_text, r.position_text)

    cenarios = {"real": "p_real", "pre": "p_pre"}
    out = {"season": SEASON, "races": [], "summary": {}}
    for rnd, g in s.groupby("round"):
        race = races[(races.season == SEASON) & (races["round"] == rnd)].iloc[0]
        item = {"round": int(rnd), "race_name": race["race_name"], "date": race["date"], "scenarios": {}}
        real_top = set(g.loc[g.position <= 10, "driver_id"])
        for key, col in cenarios.items():
            gg = g.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
            if len(gg) < len(g):          # cenário sem dados suficientes (ex.: 1ª corrida, sem histórico)
                continue
            top = gg.head(10)
            rank = {d: i + 1 for i, d in enumerate(gg.driver_id)}
            item["scenarios"][key] = {
                "hits": int(len(set(top.driver_id) & real_top)),
                "drivers": [{"driver": nome(r.driver_id), "team": equipe(r.constructor_id),
                             "prob": round(float(getattr(r, col)), 4), "grid": int(r.grid),
                             "result": resultado(r), "top10": bool(r.position <= 10)}
                            for r in top.itertuples()],
                "missed": [{"driver": nome(r.driver_id), "team": equipe(r.constructor_id),
                            "result": resultado(r), "pred_rank": rank[r.driver_id],
                            "prob": round(float(getattr(r, col)), 4), "grid": int(r.grid)}
                           for r in gg[gg.driver_id.isin(real_top - set(top.driver_id))].itertuples()],
            }
        out["races"].append(item)

    for key, col in cenarios.items():
        rs = [r for r in out["races"] if key in r["scenarios"]]
        sub = s[s["round"].isin([r["round"] for r in rs])]
        hits = sum(r["scenarios"][key]["hits"] for r in rs)
        out["summary"][key] = {
            "races": len(rs), "hits": hits, "possible": 10 * len(rs),
            "accuracy": round(float(accuracy_score(sub.target_top10, sub[col] >= .5)), 4),
            "auc": round(float(roc_auc_score(sub.target_top10, sub[col])), 4),
            "perfect": sum(r["scenarios"][key]["hits"] == 10 for r in rs),
        }
    out["last_race"] = out["races"][-1]["race_name"]
    out["last_round"] = out["races"][-1]["round"]

    with open("comparacao_previsoes.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if os.path.exists(HTML):
        html = open(HTML, encoding="utf-8").read()
        html = re.sub(r"const DATA = \{.*?\};\n", lambda m: "const DATA = " + json.dumps(out, ensure_ascii=False) + ";\n",
                      html, count=1, flags=re.S)
        open(HTML, "w", encoding="utf-8").write(html)

    for key, lab in [("real", "com grid real"), ("pre", "antes do classificatório")]:
        sm = out["summary"][key]
        print(f"{lab:26s}: {sm['hits']}/{sm['possible']} pilotos do Top 10 em {sm['races']} corridas "
              f"({sm['hits'] / sm['possible']:.1%}), acurácia {sm['accuracy']:.1%}, AUC {sm['auc']:.3f}, "
              f"{sm['perfect']} corridas com 10/10")
    return out


if __name__ == "__main__":
    main()
