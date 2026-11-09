# Classificação com Redes Neurais — Previsão de Top 10 na Fórmula 1

Projeto de Inteligência Artificial cujo objetivo é aplicar redes neurais em um problema real de classificação com dados tabulares. Em vez do MNIST sugerido no enunciado, o projeto usa dados reais de resultados de Fórmula 1 (1950–2026) para prever se um piloto termina uma corrida entre os 10 primeiros colocados.

## Pergunta do projeto

Dado o grid de largada, a equipe, o circuito e a forma recente do piloto/equipe na temporada, **o piloto termina a corrida entre os 10 primeiros colocados?**

Essa é uma classificação binária: `1` = terminou no Top 10, `0` = não terminou no Top 10 (fora dos pontos ou não completou a corrida).

## Fonte dos dados

- **API:** [jolpica-f1](https://github.com/jolpica/jolpica-f1) — sucessor open-source e comunitário da antiga Ergast Developer API, disponível em `api.jolpi.ca`.
- **Arquivos usados:** `results.csv` (26.313 linhas — um piloto por corrida, 1950 a 2026) e `races.csv` (calendário e circuitos: localização, latitude/longitude).
- **Atualização:** os dados foram complementados via a API para cobrir até o **GP do Brasil (08/11/2026, rodada 20)**, já que o dataset original (Kaggle, `fleongg/formula-1-results-and-standings-1950-present`) estava parado no GP de Mônaco (07/06/2026). A última atualização (04/11/2026) também corrigiu o calendário de 2026, que passou a ter 23 etapas com a inclusão do "Bahrain Grand Prix in Malaysia" (Sepang, rodada 16) — com isso o GP do Brasil virou a rodada 20.
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

> **Correção de 09/11/2026 — bug no cálculo das médias da temporada.** A primeira versão do notebook calculava `driver_avg_pos_season` e `constructor_avg_points_season` com `groupby(...).apply(...).reset_index(drop=True)`. Esse padrão devolve os valores na ordem dos *grupos* (piloto a piloto), não na ordem das linhas do DataFrame. Resultado: em ~99% das linhas a média atribuída era de outro piloto ou equipe — a correlação da feature com a posição real caía de 0,41 para 0,04, ou seja, virava ruído. O modelo era treinado praticamente sem informação de forma e passava a depender quase só do grid. Isso aparecia como uma "limitação": pilotos de equipes de ponta largando atrás por punição recebiam probabilidades muito baixas (no GP do Brasil 2026, Verstappen largou em 21º com 27% e terminou em 2º). A correção foi trocar por `groupby(...).transform(...)`, que mantém o alinhamento, e retreinar o modelo. Foram testadas também features extras (grid médio do piloto e do carro, diferença entre o grid e o grid habitual, largada do pit lane), mas elas não melhoraram o resultado além da correção do bug e não foram adotadas.

Colunas como `points`, `status`, `laps` e `time_millis` foram **excluídas** por só existirem depois da corrida acontecer — usá-las seria data leakage (o modelo "colaria" a resposta em vez de aprender a prever).

## Divisão treino/teste

Em vez de um split aleatório, foi usado um **split temporal**:

- **Treino:** temporadas 1950–2023 (24.915 linhas)
- **Teste:** temporadas 2024–2026 (1.398 linhas, nunca vistas durante o treino)

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

Avaliação no conjunto de teste (temporadas 2024–2026 até o GP do Brasil, nunca vistas pelo modelo), comparando o modelo atual (com a correção das features) com a versão anterior:

| Métrica | Versão anterior (com o bug) | **Versão atual** |
|---|---|---|
| Acurácia | 77,5% | **79,0%** |
| F1-score | 0,778 | **0,785** |
| AUC-ROC | 0,853 | **0,863** |

Matriz de confusão da versão atual (1.398 corridas-piloto no teste):

|  | Previsto: Fora do Top10 | Previsto: Top10 |
|---|---|---|
| **Real: Fora do Top10** | 568 | 150 |
| **Real: Top10** | 144 | 536 |

O ganho fica mais claro nas corridas recentes e nos casos difíceis:

| Recorte | Versão anterior | **Versão atual** |
|---|---|---|
| Rodadas 13–20 de 2026, com o grid real (176 corridas-piloto) | acurácia 76,1%, AUC 0,836 | **acurácia 84,1%, AUC 0,906** |
| Pilotos do Top 10 acertados nessas 8 corridas | 61 de 80 | **67 de 80** |
| GP do Brasil 2026 (grid real, 5 punições de motor) | 7 de 10 | **10 de 10** |
| Pilotos que largaram ≥5 posições atrás do habitual | acurácia ~51% | **acurácia ~63%** |

O grid de largada continua sendo o preditor mais forte (quem larga na frente tende a terminar na frente), mas com as médias da temporada calculadas corretamente a rede passa a combinar o grid com a forma real do piloto e da equipe. Repetindo o treino com 5 sementes aleatórias diferentes, a acurácia da versão corrigida ficou em 79,0% ± 0,4 p.p. e o AUC em 0,863 ± 0,002, então o ganho não depende da semente.

> **Nota sobre reprodutibilidade:** por não-determinismo de ponto flutuante do TensorFlow em CPU (paralelismo de threads, oneDNN), rodar o notebook do zero novamente pode produzir números ligeiramente diferentes destes (variação tipicamente de ±1 ponto percentual). Os valores acima são os da execução salva no notebook (09/11/2026), e são a mesma execução que gerou `f1_top10_model.keras`, `dashboard_data.json`, `dashboard.html` e `predicoes_proximas_corridas_2026.csv`.

### Previsão para corridas futuras (demonstração de uso)

Como aplicação prática, o modelo é usado para estimar a probabilidade de Top 10 nas próximas corridas de 2026, com a forma da temporada até a última corrida disputada. A previsão é feita por `prever_proximas_corridas.py`, que o notebook chama na seção 9 e que também pode ser rodado sozinho depois de cada corrida, sem retreinar o modelo.

**Previsão atual — forma até o GP do Brasil (rodada 20), modelo corrigido:**

| Corrida | Top 10 previsto (em ordem de probabilidade) | Logo atrás |
|---|---|---|
| GP de Las Vegas (rodada 21, 22/11) | Antonelli 90,0%, Russell 89,2%, Hamilton 87,7%, Leclerc 87,3%, Norris 81,8%, Verstappen 81,4%, Hadjar 80,9%, Piastri 79,7%, Lawson 59,2%, Lindblad 53,2% | Gasly 45,0%, Colapinto 38,3%, Hülkenberg 36,7% |
| GP do Catar (rodada 22, 29/11) | Antonelli 89,4%, Russell 88,6%, Hamilton 87,2%, Leclerc 86,3%, Norris 80,9%, Verstappen 80,0%, Hadjar 79,9%, Piastri 78,5%, Lawson 61,0%, Lindblad 55,2% | Gasly 48,9%, Colapinto 38,4%, Hülkenberg 37,5% |
| GP de Abu Dhabi (rodada 23, 06/12) | Antonelli 90,7%, Russell 90,2%, Hamilton 88,8%, Leclerc 88,2%, Verstappen 83,1%, Norris 82,9%, Hadjar 82,4%, Piastri 81,4%, Lawson 61,0%, Lindblad 54,8% | Gasly 48,6%, Colapinto 39,5%, Hülkenberg 34,1% |

As quatro equipes de ponta (Mercedes, Ferrari, McLaren e Red Bull) ocupam as 8 primeiras vagas com folga (78%–91%). As duas últimas ficam com a Racing Bulls (Lawson e Lindblad, 53%–61%), com Gasly logo atrás, perto de 50%. Probabilidades completas (os 22 pilotos, mais uma variante "forma recente" que usa o grid médio das últimas 5 corridas) em `predicoes_proximas_corridas_2026.csv`.

**Escalações consideradas:** Hadjar na Red Bull e Lawson na Racing Bulls (desde a rodada 15); Tsunoda fora do grid. Na Haas, Esteban Ocon foi afastado pelo resto de 2026 e o brasileiro **Rafael Câmara** estreou no GP do Brasil (mantido nas demais etapas por hipótese — ajuste `SUBSTITUICOES` no script se mudar). Pilotos com menos de 3 corridas na temporada, como Câmara, recebem as médias de grid e de posição do carro (`MIN_CORRIDAS`), para que uma corrida isolada não distorça a previsão.

**Grid real quando disponível:** depois do classificatório, a posição oficial de largada de cada piloto pode ser informada em `GRID_REAL` no script (foi assim no GP do Brasil). Sem ela, o `grid` é substituído pela média de grid do piloto na temporada — uma projeção da tendência de desempenho, não do resultado do fim de semana. O modelo prevê "Top 10 sim/não", não a ordem de chegada: as listas acima estão ordenadas por probabilidade, não por posição prevista.

**Histórico das previsões de 2026:**
- **04/11 (forma até o México, modelo anterior):** a projeção feita com dados até a Holanda tinha acertado 7 de 10 pilotos do Top 10 na Itália, 9 na Espanha e 6 no Azerbaijão.
- **07/11 (GP do Brasil com o grid real, modelo anterior):** as punições de motor (Hamilton, Verstappen e Pérez no fundo; Leclerc −10; Russell −5) derrubaram Hamilton para 24% e Verstappen para 18%. A previsão acertou 7 de 10. Verstappen terminou em 2º, Hamilton em 9º e Leclerc em 8º.
- **09/11 (modelo corrigido):** com o mesmo grid, o modelo corrigido dá 76% para Hamilton, 64% para Verstappen e 79% para Leclerc, e acerta os 10 pilotos do Top 10 do Brasil.

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
