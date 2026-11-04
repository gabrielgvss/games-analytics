# Game Sales Lab

Análise reproduzível de vendas históricas de videogames com regressão linear, acompanhada de um [simulador interativo offline](index.html). A análise acadêmica original é de **Gabriel Silveira e Marcos Reis**; a evolução deste repositório para portfólio é de **Gabriel Silveira**.

## Experimente

Abra [`index.html`](index.html) com dois cliques. O arquivo contém HTML, CSS, JavaScript, parâmetros do modelo treinado e resumos dos resultados. Não precisa de internet, servidor, Python ou conta Kaggle para funcionar. Busque entre **3.343 jogos com nome do conjunto reservado**, escolha um dos casos sugeridos ou altere os dados de um jogo. A estimativa e a explicação mudam na hora. A página apresenta primeiro o resultado em linguagem de negócio; os detalhes do método ficam opcionais.

**Interpretação:** o modelo usa notas e contagens de avaliações observadas depois do lançamento. A ferramenta explora padrões de uma base histórica, não prevê com segurança as vendas de jogos futuros. Vendas são apresentadas em milhões de unidades.

## Reproduza o estudo

Requisitos: Python 3.11+ e, para verificar o JavaScript, Node.js 18+. Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m games_analytics.build
.\.venv\Scripts\python.exe scripts\refresh_narrative.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Em macOS/Linux, ative o ambiente virtual com `source .venv/bin/activate` e use `python` nos comandos. O `build` baixa a **versão 2** do [dataset público no Kaggle](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings) via `kagglehub`, verifica seu SHA-256 e gera `artifacts/results.json` e `index.html`. Também é possível informar `--data caminho/do/arquivo.csv` para usar uma cópia local da mesma versão. O CSV não é versionado. Não use `pip install sklearn`; o pacote correto é `scikit-learn` e já consta em `pyproject.toml`.

Depois do build, execute o [notebook](games_analytics.ipynb) para reproduzir as figuras acadêmicas. O [relatório escrito](doc.md) é gerado do mesmo JSON de resultados.

## Como o modelo foi avaliado

- Divisão 80/20 com semente 42; cinco dobras de validação cruzada apenas no treino.
- Imputação, escala, categorias raras e one-hot dentro dos pipelines de cada dobra.
- Sete candidatos, incluindo baseline, regressão simples, regressões múltiplas, alvo em log e polinomial.
- Seleção por RMSE de validação com critério explícito de simplicidade; comparação dos sete no teste **após** a escolha.
- Métricas brutas iguais para todos os modelos. O corte de previsões negativas em zero é mostrado separadamente para a experiência de uso.

O HTML calcula a mesma função que o pipeline final: uma previsão de referência mais contribuições independentes por campo. É uma decomposição aritmética **exata** da regressão ajustada. Ela explica o cálculo, não estabelece causalidade. O VIF numérico, as correlações e a redundância dos indicadores categóricos são discutidos no relatório.

## Estrutura

| Local | Conteúdo |
|---|---|
| `src/games_analytics/core.py` | Dados, transformações, candidatos e avaliação |
| `src/games_analytics/build.py` | Treino, diagnósticos e exportação do modelo para HTML |
| `web/` | Fonte editável do HTML, CSS e JavaScript |
| `index.html` | Entrega final offline, autossuficiente |
| `games_analytics.ipynb`, `doc.md` | Análise acadêmica e narrativa atualizadas |
| `tests/` | Paridade entre Python e JavaScript, além de verificações de dados |

O projeto foi desenvolvido com scikit-learn. Não há API, servidor nem rastreamento de visitantes.
