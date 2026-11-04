"""
Atualiza results.csv e o calendário da temporada em races.csv a partir da API
jolpica-f1 (api.jolpi.ca), acrescentando só as corridas que ainda faltam.

Uso (dentro da pasta dataframes/):
    pip install requests
    python atualizar_dados.py            # temporada atual (2026)
    python atualizar_dados.py 2027       # outra temporada

Depois rode, na raiz do projeto:  python prever_proximas_corridas.py
"""
import csv
import sys
import time

import requests

API = "https://api.jolpi.ca/ergast/f1"
SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026


def get(path):
    for tentativa in range(5):
        r = requests.get(f"{API}/{path}", timeout=30)
        if r.status_code == 429:          # limite de requisições: espera e tenta de novo
            time.sleep(5 * (tentativa + 1))
            continue
        r.raise_for_status()
        return r.json()["MRData"]
    raise RuntimeError(f"Falha ao buscar {path}")


def f(x):
    return "" if x in (None, "") else str(float(x))


def main():
    # ---- calendário da temporada (substitui as linhas dessa temporada em races.csv) ----
    races = get(f"{SEASON}/races.json?limit=100")["RaceTable"]["Races"]
    cal = {int(R["round"]): R for R in races}
    rows = list(csv.reader(open("races.csv", encoding="utf-8", newline="")))
    out = [r for r in rows if r[0] != str(SEASON)]
    for R in races:
        c = R["Circuit"]; loc = c["Location"]
        out.append([SEASON, R["round"], R["raceName"], R["date"], R.get("time", ""), c["circuitId"],
                    c["circuitName"], loc["locality"], loc["country"], loc["lat"], loc["long"], R.get("url", "")])
    with open("races.csv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(out)
    print(f"races.csv: calendário {SEASON} com {len(races)} etapas")

    # ---- resultados que ainda não estão em results.csv ----
    feitos = {int(r[1]) for r in csv.reader(open("results.csv", encoding="utf-8", newline="")) if r[0] == str(SEASON)}
    novas = []
    for rnd in sorted(cal):
        if rnd in feitos:
            continue
        rr = get(f"{SEASON}/{rnd}/results.json?limit=100")["RaceTable"]["Races"]
        if not rr:
            break                          # corrida ainda não aconteceu
        R = rr[0]
        for x in R["Results"]:
            d, c = x["Driver"], x["Constructor"]
            t, fl = x.get("Time", {}), x.get("FastestLap", {})
            novas.append([SEASON, rnd, R["raceName"], R["date"], d["driverId"], d.get("code", ""), x["number"],
                          f"{d['givenName']} {d['familyName']}", d["nationality"], d.get("dateOfBirth", ""),
                          c["constructorId"], c["name"], c["nationality"], x["grid"], x["position"],
                          x["positionText"], f(x["points"]), x["laps"], x["status"], t.get("time", ""),
                          f(t.get("millis")), f(fl.get("rank")), f(fl.get("lap")),
                          fl.get("AverageSpeed", {}).get("speed", "")])
        print(f"  + rodada {rnd}: {R['raceName']} ({len(R['Results'])} pilotos)")
        time.sleep(0.5)
    with open("results.csv", "a", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(novas)
    print(f"results.csv: {len(novas)} linhas novas")


if __name__ == "__main__":
    main()
