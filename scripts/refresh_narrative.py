"""Regera o notebook e a análise escrita a partir dos resultados do pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "artifacts" / "results.json").read_text(encoding="utf-8"))["report"]


def f(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


cv = next(row for row in report["cv"] if row["modelo"] == report["selected"])
test = report["final_raw"]
low = report["deciles"][0]
wii = report["wii"]
spearman = report["spearman"]
vif = report["vif"]
test_table = "\n".join(f"| {row['modelo']} | {f(row['RMSE'])} | {f(row['MAE'])} | {f(row['R2'])} |" for row in report["test_models"])
vif_table = "\n".join(f"| `{row['atributo']}` | {f(row['VIF'], 2)} |" for row in vif)

doc = f"""# Game Sales Analytics

## Análise de vendas de videogames com regressão linear

**Análise acadêmica:** Gabriel Silveira e Marcos Reis. **Evolução para portfólio:** Gabriel Silveira.
**Dados:** [Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings), versão 2, 16.719 registros.
**Alvo:** `Global_Sales`, em milhões de unidades. Cada linha representa um jogo em uma plataforma.

O estudo é retrospectivo: notas e contagens de avaliações aparecem após o lançamento. A regressão descreve associações na base histórica; não estima com segurança o sucesso de um lançamento futuro.

## 1. Entendimento e preparação

As vendas regionais (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`) foram excluídas, pois formam o alvo. `Name` foi mantido apenas para análise de erros, nunca como preditor. `User_Score = tbd` é tratado como ausente. A ausência das avaliações varia com ano, gênero e volume de vendas, então descartar essas linhas poderia distorcer a amostra.

A divisão treino/teste é de 80/20 com semente 42. Imputação, escala, agrupamento de categorias raras e codificação ficam dentro do pipeline, ajustados somente no treino de cada dobra. As transformações `log1p` das contagens são determinísticas. Nenhuma variável de vendas regionais participa da modelagem.

## 2. Exploração e colinearidade

No heatmap de Spearman, `Critic_Count` e `User_Count` têm correlação **{f(spearman['critic_user_counts'], 2)}**; `Critic_Score` e `User_Score`, **{f(spearman['critic_user_scores'], 2)}** no treino. A relação entre contagens sugere que jogos com maior cobertura crítica também tendem a ter mais avaliações de usuários. As notas também compartilham informação. Isso reduz a segurança de atribuir um efeito isolado a um coeficiente, especialmente ao interpretar sinais e magnitudes como se fossem causas.

O diagnóstico de fator de inflação da variância (VIF) usa as cinco variáveis numéricas do modelo final, após imputação pela mediana aprendida no treino:

| Atributo | VIF |
|---|---:|
{vif_table}

Os VIFs numéricos não indicam colinearidade severa nesse subconjunto. A codificação one-hot com intercepto introduz redundância entre indicadores de uma mesma categoria no modelo completo. O scikit-learn ainda encontra previsões, mas coeficientes individuais de categorias não devem ser lidos como efeitos únicos e estáveis. As contribuições exibidas na interface são uma decomposição exata do cálculo ajustado, não evidência causal.

## 3. Modelos e seleção

A regressão simples usa `Critic_Count` **em escala original**, mesmo que a dispersão exploratória use escala log para visualização. O `SimpleImputer(add_indicator=True)` acrescenta uma coluna de ausência quando há faltantes; portanto, o ajuste tem um atributo substantivo e um indicador. `LinearRegression` do scikit-learn ajusta por mínimos quadrados ordinários (OLS). Os modelos múltiplos combinam atributos numéricos e categóricos; a versão polinomial acrescenta quadrados das notas e interação.

Sete candidatos foram comparados com validação cruzada de cinco dobras no treino. O modelo escolhido foi **{report['selected']}**, com RMSE {f(cv['RMSE_cv'])} ± {f(cv['RMSE_cv_dp'])}, MAE {f(cv['MAE_cv'])} e R² {f(cv['R2_cv'])}. A melhora da regressão polinomial foi inferior a 0,5% e menor que a variação entre dobras; a versão mais simples foi preferida.

## 4. Comparação final no teste

Após a escolha pela validação, cada candidato foi treinado com todo o conjunto de treino e avaliado uma vez no mesmo teste. A tabela usa previsões **brutas**, para comparação uniforme; o teste não foi usado para trocar o modelo escolhido.

| Modelo | RMSE | MAE | R² |
|---|---:|---:|---:|
{test_table}

O modelo escolhido teve RMSE de **{f(test['RMSE'])}** no teste, acima de {f(cv['RMSE_cv'])} na validação. Seu MAE foi **{f(test['MAE'])}** e o R², **{f(test['R2'])}**, versus {f(cv['R2_cv'])} na validação. Wii Sports, que caiu no teste, vendeu {f(wii['actual'],2)} milhões; a estimativa foi {f(wii['predicted'],2)}, com resíduo de {f(wii['residual'],2)}. Sem esse único registro, o RMSE bruto seria {f(wii['rmse_without'])}. Essa retirada é apenas um diagnóstico: a métrica oficial preserva o caso real.

## 5. Erros e interpretação

No primeiro decil de vendas reais, a média observada foi **{f(low['real'],3)}** milhão e a média prevista após limitar valores negativos a zero foi **{f(low['previsto'],3)}** milhão, cerca de **{f(low['previsto']/low['real'],1)} vezes** maior. O modelo erra pouco em termos absolutos nesses jogos baratos, mas muito em termos relativos. Nos blockbusters, o erro absoluto domina o RMSE.

Ocorreram **{report['negative_count']}** previsões brutas negativas (**{f(report['negative_pct'],2)}%** do teste). A interface limita essas previsões a zero, mas informa o valor bruto e distingue as métricas de teste antes e depois do corte. A devolutiva citou 15,85%; a execução reproduzível desta versão gera {f(report['negative_pct'],2)}%.

### 5.4 Coeficientes e OLS

O notebook inclui um OLS reduzido com `statsmodels` e o alvo em `log1p`, apenas como diagnóstico. Seus coeficientes e p-valores dependem da especificação e não demonstram causalidade. A correlação entre avaliações e a redundância dos indicadores categóricos no pipeline completo exigem cautela antes de dizer que um coeficiente representa o efeito próprio de um atributo. Na interface, cada contribuição é uma parcela aritmética da previsão em relação a um jogo de referência do treino.

## 6. Conclusão

A regressão múltipla supera a regressão simples em RMSE e R² no teste, mas explica apenas parte da variação. A análise mostra dois limites diferentes: superestimação relativa nos jogos de vendas mínimas e subestimação absoluta em grandes sucessos. Faltam variáveis relevantes como marketing, orçamento e força da franquia. O HTML offline serve para explorar o modelo histórico e suas limitações, não para orientar investimentos ou prever vendas futuras com precisão.
"""
(ROOT / "doc.md").write_text(doc, encoding="utf-8")

cells = []
def md(source): cells.append(nbf.v4.new_markdown_cell(source))
def code(source): cells.append(nbf.v4.new_code_cell(source))

md("""# GAME SALES ANALYTICS
### Regressão linear de vendas de videogames · estudo reproduzível

**Análise acadêmica:** Gabriel Silveira e Marcos Reis  
**Evolução para portfólio:** Gabriel Silveira  
**Dataset:** [Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings), versão 2  
**Unidade:** um jogo em uma plataforma · **Alvo:** Global_Sales (milhões de unidades)

Este notebook acompanha o pacote Python do projeto. Gere os resultados com `python -m games_analytics.build` antes de executá-lo. A análise é retrospectiva: avaliações de crítica e usuários são observadas após o lançamento.""")
md("## 1) Business Understanding\n\nQuais características da base ajudam a explicar vendas globais históricas? O objetivo é analisar padrões e limites da regressão, não prever antecipadamente o sucesso de um jogo. RMSE é a métrica principal; MAE e R² complementam a leitura.")
code("""import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.cwd() / 'src'))
from games_analytics.core import load_data, prepare, split, candidates, TARGET, NUM_LOG, SEED

sns.set_theme(style='whitegrid')
results = json.loads(Path('artifacts/results.json').read_text(encoding='utf-8'))['report']
data = prepare(load_data())
train, test = split(data)
print(f'Treino: {len(train):,} · teste reservado: {len(test):,}')""")
md("## 2) Data Understanding\n\n`tbd` é ausência de nota, não zero. Valores faltantes nas avaliações se concentram em certos períodos, gêneros e níveis de venda. Mantemos as linhas e imputamos apenas dentro das dobras de treino.")
code("""display(train[['Year_of_Release','Critic_Score','Critic_Count','User_Score','User_Count','Rating','Developer']].isna().mean().mul(100).round(1).to_frame('ausentes_%'))
display(train[TARGET].describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_frame('vendas_milhões'))""")
code("""missing = train[['Year_of_Release','Genre',TARGET,'Critic_Score','User_Score']].copy()
missing['sem_avaliacao'] = missing[['Critic_Score','User_Score']].isna().any(axis=1)
missing['periodo'] = pd.cut(missing['Year_of_Release'],bins=[1979,1989,1999,2009,2016,2020],labels=['1980–89','1990–99','2000–09','2010–16','2017+'])
missing['faixa_venda'] = pd.cut(missing[TARGET],[-np.inf,.1,.5,1,5,np.inf],labels=['até 0,1','0,1–0,5','0,5–1','1–5','acima de 5'])
fig,axes=plt.subplots(1,3,figsize=(16,4))
for ax,column,title in zip(axes,['periodo','Genre','faixa_venda'],['Por período','Por gênero','Por vendas']):
    values=missing.groupby(column,observed=True)['sem_avaliacao'].mean().mul(100).sort_values(ascending=False)
    values.plot.bar(ax=ax,color='#747df5');ax.set_title(title);ax.set_ylabel('% sem uma das avaliações');ax.tick_params(axis='x',rotation=45)
plt.tight_layout();plt.show()""")
md("### 2.1 Relações entre atributos e vendas")
code("""columns = ['Year_of_Release','Critic_Score','Critic_Count','User_Score','User_Count',TARGET]
corr = train[columns].corr(method='spearman')
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, ax=ax)
ax.set_title('Correlação de Spearman no treino')
plt.tight_layout(); plt.show()""")
md(f"""> **Leitura imediata do heatmap:** `Critic_Count` e `User_Count` se correlacionam em **{f(spearman['critic_user_counts'],2)}**; `Critic_Score` e `User_Score`, em **{f(spearman['critic_user_scores'],2)}**. Cobertura e avaliações compartilham informação. Isso não invalida previsões, mas dificulta interpretar um coeficiente isolado da regressão múltipla como efeito próprio ou causal. A seção 5.4 volta a esse ponto com VIF e o OLS reduzido.""")
code("""fig, axes = plt.subplots(1,2,figsize=(13,4.5))
sample = train.sample(min(5000,len(train)),random_state=SEED)
sns.scatterplot(data=sample,x='Critic_Count',y=TARGET,alpha=.2,s=14,ax=axes[0])
axes[0].set_yscale('log'); axes[0].set_title('Cobertura crítica e vendas · eixo y log')
score = train[['Critic_Score',TARGET]].dropna().copy()
score['faixa_nota'] = pd.cut(score['Critic_Score'],bins=[0,20,40,60,80,100],include_lowest=True,labels=['0–20','21–40','41–60','61–80','81–100'])
sns.boxplot(data=score,x='faixa_nota',y=TARGET,showfliers=False,ax=axes[1],color='#747df5')
axes[1].set_title('Vendas por faixa de nota da crítica'); axes[1].set_xlabel('Nota da crítica')
plt.tight_layout(); plt.show()""")
md("A escala log na dispersão apenas facilita a visualização. A regressão simples usa `Critic_Count` **cru**, sem log.")
md("## 3) Data Preparation\n\nA divisão 80/20 (`random_state=42`) antecede qualquer ajuste. Vendas regionais e `Name` ficam fora dos preditores. O pacote define imputação, escala e one-hot dentro dos pipelines; cada dobra aprende seus próprios parâmetros.")
code("""assert not {'NA_Sales','EU_Sales','JP_Sales','Other_Sales','Name'} & set(NUM_LOG + ['Platform','Genre','Rating','Publisher','Developer'])
print('Pipeline compartilhado no pacote: src/games_analytics/core.py')""")
md("## 4) Modeling\n\nComparamos baseline, regressão simples, três regressões múltiplas, alvo em log e polinomial. `LinearRegression` do scikit-learn ajusta os coeficientes por mínimos quadrados ordinários (OLS). A regressão simples usa uma variável substantiva, `Critic_Count`, mas a imputação cria também um indicador de ausência: duas colunas efetivas quando há faltantes.")
code("""cv = pd.DataFrame(results['cv'])
display(cv[['modelo','RMSE_cv','RMSE_cv_dp','MAE_cv','R2_cv']].round(4))
fig,ax=plt.subplots(figsize=(10,5))
ax.barh(cv['modelo'],cv['RMSE_cv'],xerr=cv['RMSE_cv_dp'],color='#747df5',capsize=3)
ax.invert_yaxis(); ax.set_xlabel('RMSE médio ± desvio entre dobras'); ax.set_title('Validação cruzada · cinco dobras')
plt.tight_layout(); plt.show()""")
md(f"**Escolha antes do teste:** {report['selected']}. O ganho de RMSE da polinomial ficou abaixo de 0,5% e menor que a variação entre dobras; a versão mais simples foi preferida.")
md("## 5) Evaluation\n\nO teste reservado foi aberto somente após a seleção. Todos os sete candidatos foram então ajustados no treino inteiro e comparados no mesmo teste. As métricas abaixo usam previsão bruta, sem corte em zero.")
code("""comparison = pd.DataFrame(results['test_models'])
display(comparison.round(4))
fig,ax=plt.subplots(figsize=(10,5))
colors=['#ff6e54' if name==results['selected'] else '#8589ac' for name in comparison['modelo']]
ax.barh(comparison['modelo'],comparison['RMSE'],color=colors)
ax.invert_yaxis(); ax.set_xlabel('RMSE no teste · milhões de unidades'); ax.set_title('Todos os modelos no teste')
plt.tight_layout(); plt.show()""")
md(f"""### 5.1 Validação versus teste

O modelo escolhido teve RMSE **{f(cv['RMSE_cv'])}** na validação e **{f(test['RMSE'])}** no teste; MAE **{f(cv['MAE_cv'])}** versus **{f(test['MAE'])}**, e R² **{f(cv['R2_cv'])}** versus **{f(test['R2'])}**. Wii Sports caiu no teste com {f(wii['actual'],2)} milhões reais, {f(wii['predicted'],2)} previstos e resíduo {f(wii['residual'],2)}. Se retirado apenas para diagnóstico, o RMSE bruto cairia a {f(wii['rmse_without'])}; a métrica oficial conserva esse caso.""")
code("""selected, columns = candidates()[results['selected']]
selected.fit(train[columns],train[TARGET])
raw = selected.predict(test[columns])
clipped = np.clip(raw,0,None)
errors = test[['Name','Platform','Genre',TARGET]].copy()
errors['previsto']=clipped; errors['residuo']=test[TARGET]-clipped
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].scatter(clipped,errors['residuo'],alpha=.22,s=13)
axes[0].axhline(0,color='black',ls='--'); axes[0].set(xlabel='Previsto',ylabel='Real − previsto',title='Resíduos no teste')
axes[1].scatter(test[TARGET],clipped,alpha=.22,s=13)
axes[1].plot([0,85],[0,85],ls='--',color='black'); axes[1].set(xlabel='Real',ylabel='Previsto',title='Real versus previsto')
plt.tight_layout(); plt.show()
display(errors.assign(erro_abs=errors['residuo'].abs()).nlargest(10,'erro_abs'))""")
md("### 5.2 Erro por decil e previsões negativas")
code("""deciles = pd.DataFrame(results['deciles'])
display(deciles.rename(columns={'real':'média real','previsto':'média prevista','n':'registros'}).round(4))
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(np.arange(1,len(deciles)+1),deciles['real'],marker='o',label='Real')
ax.plot(np.arange(1,len(deciles)+1),deciles['previsto'],marker='o',label='Previsto')
ax.set_yscale('log');ax.set(xlabel='Decil de venda real',ylabel='Milhões · escala log',title='Médias por decil')
ax.legend();plt.tight_layout();plt.show()""")
md(f"""> No primeiro decil, a venda média real foi **{f(low['real'],3)}** e a prevista, **{f(low['previsto'],3)}** milhão: quase **{f(low['previsto']/low['real'],1)} vezes** maior. O erro absoluto é pequeno, mas o relativo é alto. O modelo também subestima blockbusters. Houve **{report['negative_count']}** previsões negativas (**{f(report['negative_pct'],2)}%**), impossíveis para vendas; o corte em zero é apresentado separadamente. A devolutiva citou 15,85%, enquanto a execução desta versão produz {f(report['negative_pct'],2)}%.""")
md("### 5.3 Erro por grupo")
code("""errors['erro_abs']=errors['residuo'].abs()
errors['faixa']=pd.cut(errors[TARGET],[-np.inf,.1,.5,1,5,np.inf],labels=['até 0,1','0,1–0,5','0,5–1','1–5','acima de 5'])
for column in ['faixa','Genre','Platform']:
    display(errors.groupby(column,observed=True).agg(n=(TARGET,'size'),MAE=('erro_abs','mean')).query('n >= 20').sort_values('MAE',ascending=False).head(12))""")
code("""grouped=errors.groupby(['Genre','faixa'],observed=True).agg(MAE=('erro_abs','mean'),n=('erro_abs','size')).reset_index()
grouped.loc[grouped['n']<5,'MAE']=np.nan
matrix=grouped.pivot(index='Genre',columns='faixa',values='MAE')
fig,ax=plt.subplots(figsize=(9,6));sns.heatmap(matrix,annot=True,fmt='.2f',cmap='magma',ax=ax)
ax.set_title('MAE por gênero e faixa de vendas · células com menos de 5 jogos omitidas')
plt.tight_layout();plt.show()""")
md("### 5.4 OLS reduzido e colinearidade\n\nO VIF abaixo verifica as variáveis numéricas do modelo final no treino, após imputação. Valores modestos nesse subconjunto não eliminam a dependência entre indicadores one-hot da regressão completa.")
code("""display(pd.DataFrame(results['vif']).round(3))
imputer=SimpleImputer(strategy='median'); scaler=StandardScaler()
x_train=scaler.fit_transform(imputer.fit_transform(train[NUM_LOG]))
ols=sm.OLS(np.log1p(train[TARGET]),sm.add_constant(x_train)).fit()
display(pd.DataFrame({'atributo':['constante']+NUM_LOG,'coeficiente':ols.params.values,'p_valor':ols.pvalues.values}).round(4))
print(f'R² ajustado do OLS reduzido no treino: {ols.rsquared_adj:.4f}')""")
code("""fig,axes=plt.subplots(1,2,figsize=(12,4.5))
axes[0].scatter(ols.fittedvalues,ols.resid,alpha=.18,s=12)
axes[0].axhline(0,color='black',ls='--');axes[0].set(xlabel='Ajustado em log',ylabel='Resíduo',title='Resíduos do OLS reduzido')
sm.qqplot(ols.resid,line='45',fit=True,ax=axes[1]);axes[1].set_title('Q–Q dos resíduos')
plt.tight_layout();plt.show()""")
md("**Leitura dos coeficientes:** as contagens e as notas compartilham informação; mudanças no conjunto de atributos podem alterar coeficientes individuais. O alvo do OLS reduzido está em log, e as variáveis foram padronizadas. P-valores e sinais não demonstram causalidade. A interface usa contribuições aditivas exatas do modelo preditivo, mas também não representa causas.")
md("## 6) Conclusão\n\nA regressão múltipla supera a simples em RMSE e R² no teste. Há erros de natureza diferente: superestimação relativa nas vendas mínimas e subestimação absoluta dos grandes sucessos. O modelo explica apenas parte da variação; faltam orçamento, marketing e força da franquia. Use o HTML offline para explorar o cálculo e seus limites, não como previsão segura de novos lançamentos.")

notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}})
nbf.write(notebook, ROOT / "games_analytics.ipynb")
print("Atualizados: doc.md e games_analytics.ipynb")
