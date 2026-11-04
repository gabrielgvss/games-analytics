# Game Sales Analytics

## Análise de vendas de videogames com regressão linear

**Análise acadêmica:** Gabriel Silveira e Marcos Reis. **Evolução para portfólio:** Gabriel Silveira.
**Dados:** [Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings), versão 2, 16.719 registros.
**Alvo:** `Global_Sales`, em milhões de unidades. Cada linha representa um jogo em uma plataforma.

O estudo é retrospectivo: notas e contagens de avaliações aparecem após o lançamento. A regressão descreve associações na base histórica; não estima com segurança o sucesso de um lançamento futuro.

## 1. Entendimento e preparação

As vendas regionais (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`) foram excluídas, pois formam o alvo. `Name` foi mantido apenas para análise de erros, nunca como preditor. `User_Score = tbd` é tratado como ausente. A ausência das avaliações varia com ano, gênero e volume de vendas, então descartar essas linhas poderia distorcer a amostra.

A divisão treino/teste é de 80/20 com semente 42. Imputação, escala, agrupamento de categorias raras e codificação ficam dentro do pipeline, ajustados somente no treino de cada dobra. As transformações `log1p` das contagens são determinísticas. Nenhuma variável de vendas regionais participa da modelagem.

## 2. Exploração e colinearidade

No heatmap de Spearman, `Critic_Count` e `User_Count` têm correlação **0,58**; `Critic_Score` e `User_Score`, **0,53** no treino. A relação entre contagens sugere que jogos com maior cobertura crítica também tendem a ter mais avaliações de usuários. As notas também compartilham informação. Isso reduz a segurança de atribuir um efeito isolado a um coeficiente, especialmente ao interpretar sinais e magnitudes como se fossem causas.

O diagnóstico de fator de inflação da variância (VIF) usa as cinco variáveis numéricas do modelo final, após imputação pela mediana aprendida no treino:

| Atributo | VIF |
|---|---:|
| `Year_of_Release` | 1,08 |
| `Critic_Score` | 1,68 |
| `Log_Critic_Count` | 1,47 |
| `User_Score` | 1,34 |
| `Log_User_Count` | 1,48 |

Os VIFs numéricos não indicam colinearidade severa nesse subconjunto. A codificação one-hot com intercepto introduz redundância entre indicadores de uma mesma categoria no modelo completo. O scikit-learn ainda encontra previsões, mas coeficientes individuais de categorias não devem ser lidos como efeitos únicos e estáveis. As contribuições exibidas na interface são uma decomposição exata do cálculo ajustado, não evidência causal.

## 3. Modelos e seleção

A regressão simples usa `Critic_Count` **em escala original**, mesmo que a dispersão exploratória use escala log para visualização. O `SimpleImputer(add_indicator=True)` acrescenta uma coluna de ausência quando há faltantes; portanto, o ajuste tem um atributo substantivo e um indicador. `LinearRegression` do scikit-learn ajusta por mínimos quadrados ordinários (OLS). Os modelos múltiplos combinam atributos numéricos e categóricos; a versão polinomial acrescenta quadrados das notas e interação.

Sete candidatos foram comparados com validação cruzada de cinco dobras no treino. O modelo escolhido foi **Linear múltipla — contagens em log**, com RMSE 1,207 ± 0,061, MAE 0,500 e R² 0,256. A melhora da regressão polinomial foi inferior a 0,5% e menor que a variação entre dobras; a versão mais simples foi preferida.

## 4. Comparação final no teste

Após a escolha pela validação, cada candidato foi treinado com todo o conjunto de treino e avaliado uma vez no mesmo teste. A tabela usa previsões **brutas**, para comparação uniforme; o teste não foi usado para trocar o modelo escolhido.

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear múltipla — contagens em log | 1,797 | 0,530 | 0,218 |
| Regressão polinomial — grau 2 | 1,798 | 0,527 | 0,217 |
| Linear múltipla — Publisher/Developer | 1,813 | 0,522 | 0,203 |
| Linear — alvo em log | 1,845 | 0,420 | 0,175 |
| Linear múltipla — base | 1,939 | 0,543 | 0,089 |
| Linear simples — Critic_Count | 1,973 | 0,566 | 0,057 |
| Baseline — mediana | 2,070 | 0,492 | -0,038 |

O modelo escolhido teve RMSE de **1,797** no teste, acima de 1,207 na validação. Seu MAE foi **0,530** e o R², **0,218**, versus 0,256 na validação. Wii Sports, que caiu no teste, vendeu 82,53 milhões; a estimativa foi 6,33, com resíduo de 76,20. Sem esse único registro, o RMSE bruto seria 1,222. Essa retirada é apenas um diagnóstico: a métrica oficial preserva o caso real.

## 5. Erros e interpretação

No primeiro decil de vendas reais, a média observada foi **0,020** milhão e a média prevista após limitar valores negativos a zero foi **0,201** milhão, cerca de **9,8 vezes** maior. O modelo erra pouco em termos absolutos nesses jogos baratos, mas muito em termos relativos. Nos blockbusters, o erro absoluto domina o RMSE.

Ocorreram **539** previsões brutas negativas (**16,12%** do teste). A interface limita essas previsões a zero, mas informa o valor bruto e distingue as métricas de teste antes e depois do corte. A devolutiva citou 15,85%; a execução reproduzível desta versão gera 16,12%.

### 5.4 Coeficientes e OLS

O notebook inclui um OLS reduzido com `statsmodels` e o alvo em `log1p`, apenas como diagnóstico. Seus coeficientes e p-valores dependem da especificação e não demonstram causalidade. A correlação entre avaliações e a redundância dos indicadores categóricos no pipeline completo exigem cautela antes de dizer que um coeficiente representa o efeito próprio de um atributo. Na interface, cada contribuição é uma parcela aritmética da previsão em relação a um jogo de referência do treino.

## 6. Conclusão

A regressão múltipla supera a regressão simples em RMSE e R² no teste, mas explica apenas parte da variação. A análise mostra dois limites diferentes: superestimação relativa nos jogos de vendas mínimas e subestimação absoluta em grandes sucessos. Faltam variáveis relevantes como marketing, orçamento e força da franquia. O HTML offline serve para explorar o modelo histórico e suas limitações, não para orientar investimentos ou prever vendas futuras com precisão.
