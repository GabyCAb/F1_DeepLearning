# Classificação com Redes Neurais — Previsão de Top 10 na Fórmula 1

Projeto de Inteligência Artificial cujo objetivo é aplicar redes neurais em um problema real de classificação com dados tabulares. Em vez do MNIST sugerido no enunciado, o projeto usa dados reais de resultados de Fórmula 1 (1950–2026) para prever se um piloto termina uma corrida entre os 10 primeiros colocados.

## Pergunta do projeto

Dado o grid de largada, a equipe, o circuito e a forma recente do piloto/equipe na temporada, **o piloto termina a corrida entre os 10 primeiros colocados?**

Essa é uma classificação binária: `1` = terminou no Top 10, `0` = não terminou no Top 10 (fora dos pontos ou não completou a corrida).

## Fonte dos dados

- **API:** [jolpica-f1](https://github.com/jolpica/jolpica-f1) — sucessor open-source e comunitário da antiga Ergast Developer API, disponível em `api.jolpi.ca`.
- **Arquivos usados:** `results.csv` (26.137 linhas — um piloto por corrida, 1950 a 2026) e `races.csv` (calendário e circuitos: localização, latitude/longitude).
- **Atualização:** os dados foram complementados via a API para cobrir até o **GP da Cidade do México (01/11/2026, rodada 19)**, já que o dataset original (Kaggle, `fleongg/formula-1-results-and-standings-1950-present`) estava parado no GP de Mônaco (07/06/2026). A última atualização (04/11/2026) também corrigiu o calendário de 2026, que passou a ter 23 etapas com a inclusão do "Bahrain Grand Prix in Malaysia" (Sepang, rodada 16) — com isso o GP do Brasil virou a rodada 20.
- **Licença:** CC BY-NC-SA 4.0 (uso não comercial, com atribuição) — projeto de fins educacionais.

## Por que "Top 10" e não "pontuação"

O sistema de pontos da F1 mudou várias vezes ao longo da história (por décadas só os 6 primeiros pontuavam; a pontuação até o 10º lugar só existe desde 2010). Por isso o alvo foi definido como **"terminar entre os 10 primeiros colocados"** — um critério válido em qualquer época — em vez de "pontuar", que dependeria da regra vigente em cada temporada.

## Engenharia de features (evitando data leakage)

Um cuidado central do projeto foi usar apenas informações disponíveis **antes** do resultado da corrida, para não vazar dados do futuro para o modelo:

| Feature | Tipo | Descrição |
|---|---|---|
| `grid` | numérica | Posição de largada (resultado do qualifying) |
| `lat`, `long` | numérica | Coordenadas do circuito |
| `season` | numérica | Temporada (captura evolução de regras e carros) |
| `driver_avg_pos_season` | numérica | Média de posição do piloto na temporada, calculada só com corridas **anteriores** à atual |
| `constructor_avg_points_season` | numérica | Média de pontos da equipe na temporada, mesma lógica |
| `constructor_id` | categórica | Equipe |
| `circuit_id` | categórica | Circuito |
| `driver_nationality` | categórica | Nacionalidade do piloto |

Colunas como `points`, `status`, `laps` e `time_millis` foram **excluídas** por só existirem depois da corrida acontecer — usá-las seria data leakage (o modelo "colaria" a resposta em vez de aprender a prever).

## Divisão treino/teste

Em vez de um split aleatório, foi usado um **split temporal**:

- **Treino:** temporadas 1950–2023 (24.915 linhas)
- **Teste:** temporadas 2024–2026 (1.222 linhas, nunca vistas durante o treino)

Essa escolha simula o cenário real de uso do modelo — prever corridas futuras a partir do histórico — e evita qualquer contaminação temporal entre treino e teste.

## Pré-processamento

- **Variáveis numéricas:** padronização com `StandardScaler` (necessário para a rede convergir bem).
- **Variáveis categóricas:** one-hot encoding com `OneHotEncoder(handle_unknown='ignore')` — o parâmetro `handle_unknown` é essencial porque equipes/circuitos novos que só aparecem no conjunto de teste (ex.: Cadillac F1 Team, que entrou em 2026) não quebram o pipeline.

## Arquitetura da rede neural

MLP (*Multilayer Perceptron*) construída em Keras/TensorFlow:

```
Input (325 features após one-hot)
  → Dense(64, relu) → Dropout(0.3)
  → Dense(32, relu) → Dropout(0.2)
  → Dense(16, relu)
  → Dense(1, sigmoid)
```

- **Otimizador:** Adam (learning rate 1e-3)
- **Função de perda:** Binary Crossentropy
- **Regularização:** Dropout + `EarlyStopping` monitorando o AUC de validação (paciência de 8 épocas, restaurando os melhores pesos)
- **Validação:** os últimos 15% do conjunto de treino (já ordenado por temporada/rodada), correspondendo às temporadas mais recentes dentro do período de treino

## Métricas utilizadas e por quê

| Métrica | O que mede | Por que foi usada aqui |
|---|---|---|
| **Acurácia** | Proporção geral de previsões corretas | Métrica intuitiva, mas isolada pode enganar — o alvo é razoavelmente balanceado (~45% Top10), então aqui é informativa |
| **F1-score** | Média harmônica entre precisão e recall | Importante porque equilibra falsos positivos e falsos negativos, além da simples acurácia |
| **AUC-ROC** | Capacidade do modelo de separar as duas classes em todos os limiares de decisão | Métrica mais robusta a desbalanceamento e independente do limiar de corte (0.5) escolhido |
| **Matriz de confusão** | Detalha acertos/erros por classe (verdadeiro/falso positivo/negativo) | Permite ver se o modelo erra mais para um lado (ex.: prevê Top10 demais ou de menos) |
| **Curvas de treino (loss/acurácia por época)** | Evolução do aprendizado | Ajuda a diagnosticar overfitting (quando a curva de validação piora enquanto a de treino melhora) |

## Resultados obtidos

Avaliação no conjunto de teste (temporadas 2024–2026, nunca vistas pelo modelo):

| Métrica | Valor |
|---|---|
| Acurácia | 77,8% |
| F1-score | 0,783 |
| AUC-ROC | 0,853 |

Matriz de confusão (1.222 corridas-piloto no teste):

|  | Previsto: Fora do Top10 | Previsto: Top10 |
|---|---|---|
| **Real: Fora do Top10** | 462 | 160 |
| **Real: Top10** | 111 | 489 |

O grid de largada é, isoladamente, um forte preditor (quem larga na frente tende a terminar na frente), mas a rede também aprende padrões relevantes de equipe, circuito e forma recente do piloto/equipe na temporada — o ganho de desempenho em relação a um modelo trivial confirma que essas features agregam informação real.

> **Nota sobre reprodutibilidade:** por não-determinismo de ponto flutuante do TensorFlow em CPU (paralelismo de threads, oneDNN), rodar o notebook do zero novamente pode produzir números ligeiramente diferentes destes (variação tipicamente de ±1 ponto percentual). Os valores acima são os da execução salva no notebook, e são a mesma execução que gerou `dashboard_data.json`, `dashboard.html` e `predicoes_proximas_corridas_2026.csv` — os três artefatos são consistentes entre si porque vêm de uma única execução, exportada ao final do notebook (seção 10), em vez de retreinos separados.

### Previsão para corridas futuras (demonstração de uso)

Como aplicação prática, o modelo salvo é usado para estimar a probabilidade de Top 10 nas próximas corridas de 2026, com a forma da temporada até a última corrida disputada. A previsão é feita pelo script `prever_proximas_corridas.py`, que reproduz exatamente o pré-processamento do notebook (as probabilidades antigas foram reproduzidas com diferença < 1e-7) e carrega `f1_top10_model.keras` sem retreinar.

**Atualização de 04/11/2026 — forma até o GP do México (rodada 19):**

| Corrida | Top 10 previsto (em ordem de probabilidade) | Disputa pela 10ª vaga |
|---|---|---|
| GP do Brasil (rodada 20, 08/11) | Hamilton 80,5%, Leclerc 80,2%, Russell 79,6%, Verstappen 78,5%, Norris 77,8%, Antonelli 77,7%, Piastri 76,5%, Hadjar 71,2%, Gasly 59,6%, Hülkenberg 56,8% | Lawson 53,3%, Lindblad 52,5%, Colapinto 49,3% |
| GP de Las Vegas (rodada 21, 22/11) | Leclerc, Russell, Hamilton, Antonelli, Norris, Verstappen, Piastri, Hadjar, Gasly, Hülkenberg | Lindblad 54%, Lawson 53% |
| GP do Catar (rodada 22, 29/11) | Leclerc, Russell, Hamilton, Norris, Piastri, Antonelli, Verstappen, Hadjar, Hülkenberg, Gasly | Lawson 40%, Lindblad 38% |
| GP de Abu Dhabi (rodada 23, 06/12) | Leclerc, Russell, Hamilton, Norris, Piastri, Antonelli, Verstappen, Hadjar, Gasly, Hülkenberg | Lawson 45%, Lindblad 45% |

Probabilidades completas (todos os 22 pilotos, mais uma variante "forma recente" que usa o grid médio das últimas 5 corridas) em `predicoes_proximas_corridas_2026.csv`.

As quatro equipes de ponta (Mercedes, Ferrari, McLaren, Red Bull) ocupam as 8 primeiras vagas com folga (65%–82%); as duas últimas são uma disputa aberta entre Alpine, Audi e Racing Bulls, quase sempre na faixa de 45%–60%, ou seja, praticamente cara ou coroa.

**Mudanças de escalação consideradas:** Hadjar no lugar de Lawson na Red Bull e Lawson na Racing Bulls (desde a rodada 15); Tsunoda fora do grid. Na Haas, Esteban Ocon foi afastado pelo resto de 2026 e o brasileiro **Rafael Câmara** estreia no GP do Brasil (a Haas só confirmou Câmara para o Brasil até 03/11; ele foi mantido nas demais etapas por hipótese — ajuste `SUBSTITUICOES` no script se mudar). Como estreante, Câmara não tem histórico na temporada, então recebe a média de grid e de posição do carro da Haas em 2026.

**Como as previsões anteriores se saíram:** a projeção feita com dados até a Holanda acertou 7 de 10 pilotos no Top 10 da Itália, 9 de 10 na Espanha e 6 de 10 no Azerbaijão. Avaliado com o grid real nas 7 corridas novas (rodadas 13–19, 154 corridas-piloto), o modelo teve acurácia de 79,2% e AUC de 0,849 — em linha com o teste original. No conjunto de teste completo atualizado (2024–2026, 1.376 linhas): acurácia de 77,6% e AUC de 0,854.

**Limitação importante:** como o treino classificatório dessas corridas ainda não aconteceu, o `grid` real foi substituído pela média de grid do piloto na temporada — ou seja, é uma projeção da tendência de desempenho, não uma previsão do resultado do fim de semana. O modelo também prevê "Top 10 sim/não", não a ordem de chegada: a lista acima está ordenada por probabilidade, não por posição prevista. O ideal é rodar o script de novo com o grid real assim que o qualifying de cada corrida acontecer.

> As métricas de treino exibidas no dashboard (77,8% / 0,853, matriz de confusão, curvas) continuam sendo as da execução original do notebook, já que o modelo não foi retreinado.

## Possíveis melhorias futuras

- Incluir dados de qualifying (tempo de volta, não só posição final do treino classificatório) e de pit stops, que não estavam disponíveis nesta versão do dataset.
- Testar outras arquiteturas (mais camadas, regularização L2, tuning de hiperparâmetros com Keras Tuner ou Optuna).
- Restringir a análise à era moderna (2010 em diante, quando a regra do Top10 pontuando é a mesma de hoje) para um recorte mais homogêneo de regras e formato de grid.
- Comparar com baselines simples (regressão logística, árvore de decisão, XGBoost) para quantificar o ganho real de usar uma rede neural em vez de um modelo mais simples.
- Recalcular as previsões de corridas futuras assim que o grid real de cada qualifying estiver disponível.

## Arquivos do projeto

- `dataframes/results.csv` — resultados de cada piloto em cada corrida (1950–2026)
- `dataframes/races.csv` — calendário de corridas e dados de circuitos
- `dataframes/README.md` — documentação original do dataset (fonte, licença, colunas)
- `F1_Classificacao_Top10.ipynb` — notebook completo do projeto (pipeline, modelo, avaliação e previsões), já executado com todas as saídas e gráficos embutidos
- `dashboard_data.json` — métricas, curvas e previsões exportadas pelo notebook (fonte dos dois artefatos abaixo)
- `dashboard.html` — painel visual interativo com os resultados (métricas, curvas de treino, matriz de confusão, ROC e previsões), abre em qualquer navegador sem precisar de servidor ou instalação
- `predicoes_proximas_corridas_2026.csv` — probabilidades de Top10 previstas para as próximas corridas (hoje: Brasil, Las Vegas, Catar e Abu Dhabi)
- `prever_proximas_corridas.py` — recalcula as previsões a partir do modelo salvo e atualiza o CSV, o `dashboard_data.json` e o `dashboard.html`
- `dataframes/atualizar_dados.py` — busca na API jolpica-f1 as corridas que faltam e o calendário atualizado
- `f1_top10_model.keras` — modelo treinado salvo (pode ser recarregado com `keras.models.load_model(...)` sem precisar retreinar)
- `README_projeto.md` — este documento

## Como visualizar os resultados

A forma mais rápida é abrir `dashboard.html` direto no navegador (duplo clique no arquivo) — painel visual com todos os resultados, sem precisar instalar nada.

Para o detalhamento do código e da análise, abra `F1_Classificacao_Top10.ipynb` em Jupyter, VSCode (com a extensão Jupyter) ou Google Colab — os resultados (métricas, gráficos, previsões) já estão salvos dentro do arquivo, sem necessidade de executar nada. Para rodar o notebook do zero, é necessário Python 3.11+ com `pandas`, `numpy`, `scikit-learn`, `tensorflow` e `matplotlib` instalados (ver `requirements.txt`).

## Como manter os dados atualizados

Este projeto usa dados até uma corrida específica de 2026 (ver seção "Fonte dos dados"), mas a temporada continua avançando. Se você clonou este repositório e quer manter os dados e as previsões atualizados a cada corrida, há três caminhos possíveis:

**1. Atualização manual, corrida a corrida.** Depois de cada GP, é possível buscar o resultado mais recente na API [jolpica-f1](https://api.jolpi.ca) e adicionar a linha correspondente em `results.csv`/`races.csv`. É a forma mais simples, mas exige repetir o processo manualmente a cada fim de semana de corrida.

**2. Tarefa agendada (para quem usa Claude/Cowork).** Quem tiver acesso a um agente com tarefas agendadas pode configurar uma rotina recorrente (ex.: semanal, ou logo após cada corrida) que busca os resultados novos na API, atualiza os CSVs e roda a etapa de inferência para gerar previsões frescas das próximas corridas — tudo de forma automática, sem intervenção manual.

**3. Rodar os scripts localmente (opção mais independente).** Dentro de `dataframes/`, rode `python atualizar_dados.py` (precisa de `pip install requests`): ele acrescenta só as corridas que ainda faltam e atualiza o calendário da temporada. Depois, na raiz do projeto, rode `python prever_proximas_corridas.py` para gerar as previsões novas. Rodar esse script na sua própria máquina é o caminho mais robusto para manter os dados sempre em dia, já que não depende de nenhum serviço de terceiros além da própria API — e pode ser agendado para rodar sozinho (por exemplo, via Agendador de Tarefas do Windows) logo após o horário de cada corrida.

**Importante: o modelo treinado não precisa ser retreinado a cada atualização.** O `f1_top10_model.keras` já capturou os padrões gerais de grid, equipe, circuito e forma dos pilotos — o que precisa ser recalculado a cada corrida nova são apenas os *features* de forma da temporada (`driver_avg_pos_season`, `constructor_avg_points_season`), que são médias até a última corrida disputada. Com os dados atualizados, basta recarregar o modelo salvo (`keras.models.load_model('f1_top10_model.keras')`) e rodar a etapa de inferência novamente para obter previsões atualizadas das próximas corridas, sem precisar treinar do zero.

## Sobre publicar este projeto

Antes de subir para um repositório público no GitHub, vale ter em mente:

- **Licença dos dados:** o dataset (via jolpica-f1/Ergast) é CC BY-NC-SA 4.0 — uso não comercial, com atribuição e compartilhamento pela mesma licença. Adequado para um projeto de portfólio/acadêmico, mas mencione a fonte e a licença no repositório (este README já faz isso) e não o apresente como produto comercial.
- **Marca:** "Fórmula 1", "F1" e os nomes de equipes/pilotos são marcas de terceiros; este projeto não é afiliado, endossado ou associado à Formula One Group. Bom incluir esse aviso no README do repositório.
- **`dashboard.html` é standalone:** funciona sozinho, sem depender do link privado do Artifact gerado durante o desenvolvimento — é esse arquivo (não o link) que deve ir para o repositório.
- **Ambiente:** adicione um `requirements.txt` (ou `environment.yml`) fixando as versões usadas (`tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`) para que o notebook seja reprodutível por outra pessoa.
