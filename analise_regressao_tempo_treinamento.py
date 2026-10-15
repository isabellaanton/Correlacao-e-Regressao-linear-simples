
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

# ------------------------------------------------------------------
# 0. BASE DE DADOS
# ------------------------------------------------------------------
dados = [
    (79.7, 93, 23, 149), (73.9, 115, 14, 145), (66.8, 109, 7, 145),
    (66.3, 71, 21, 144), (95.7, 114, 25, 213), (89.3, 118, 25, 145),
    (80.3, 117, 24, 119), (53.5, 28, 12, 233), (70.5, 65, 20, 218),
    (68.5, 81, 25, 116), (48.2, 48, 17, 88), (58.1, 58, 24, 92),
    (66.5, 83, 17, 81), (65.8, 100, 8, 149), (64.7, 78, 14, 159),
    (56.6, 37, 17, 172), (71.3, 88, 11, 221), (87.4, 108, 19, 229),
    (51.4, 42, 13, 152), (74.7, 74, 24, 161), (38.0, 54, 9, 65),
    (77.5, 111, 18, 101), (49.6, 26, 6, 240), (65.1, 68, 7, 227),
]
df = pd.DataFrame(dados, columns=["Tempo", "Registros", "Variaveis", "Estimadores"])
df.index = range(1, len(df) + 1)
df.index.name = "Exp"

Y = "Tempo"
XS = ["Registros", "Variaveis", "Estimadores"]

# ------------------------------------------------------------------
# 1. ANÁLISE EXPLORATÓRIA
# ------------------------------------------------------------------
# a) Estatísticas descritivas
print("=" * 70)
print("1a) ESTATÍSTICAS DESCRITIVAS")
print("=" * 70)
print(df.describe().round(3).to_string())
print("\nCoeficiente de variação (%):")
print((df.std() / df.mean() * 100).round(2).to_string())

# b) Matriz de correlação
print("\n" + "=" * 70)
print("1b) MATRIZ DE CORRELAÇÃO (Pearson)")
print("=" * 70)
corr = df.corr()
print(corr.round(4).to_string())
mais_corr = corr[Y].drop(Y).abs().idxmax()
print(f"\nMaior correlação simples com {Y}: {mais_corr} (r = {corr.loc[mais_corr, Y]:.4f})")

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Matriz de correlação")
plt.tight_layout()
plt.savefig("fig_correlacao.png", dpi=150)

# c) Gráficos de dispersão
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, x in zip(axes, XS):
    sns.regplot(x=df[x], y=df[Y], ax=ax, ci=None,
                scatter_kws={"s": 45}, line_kws={"color": "red"})
    ax.set_title(f"{Y} × {x}  (r = {corr.loc[x, Y]:.3f})")
plt.tight_layout()
plt.savefig("fig_dispersao.png", dpi=150)

# ------------------------------------------------------------------
# 2. MODELO DE REGRESSÃO LINEAR MÚLTIPLA
# ------------------------------------------------------------------
X = sm.add_constant(df[XS])
modelo = sm.OLS(df[Y], X).fit()

print("\n" + "=" * 70)
print("2) MODELO DE REGRESSÃO LINEAR MÚLTIPLA")
print("=" * 70)
print(modelo.summary())

b = modelo.params
print("\nEquação estimada:")
print(f"Tempo = {b['const']:.4f} + {b['Registros']:.4f}·Registros "
      f"+ {b['Variaveis']:.4f}·Variaveis + {b['Estimadores']:.4f}·Estimadores")

# ------------------------------------------------------------------
# 3. QUALIDADE DO AJUSTE
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("3) QUALIDADE DO AJUSTE")
print("=" * 70)
print(f"R²          = {modelo.rsquared:.4f}")
print(f"R² ajustado = {modelo.rsquared_adj:.4f}")
print(f"Diferença   = {modelo.rsquared - modelo.rsquared_adj:.4f}")
print(f"Erro-padrão residual (RMSE ajustado) = {np.sqrt(modelo.mse_resid):.4f}")

# ------------------------------------------------------------------
# 4. SIGNIFICÂNCIA ESTATÍSTICA (alpha = 0,05)
# ------------------------------------------------------------------
alpha = 0.05
print("\n" + "=" * 70)
print("4) SIGNIFICÂNCIA ESTATÍSTICA (α = 0,05)")
print("=" * 70)
print(f"Teste F: F = {modelo.fvalue:.3f} | p-valor = {modelo.f_pvalue:.3e}")
print("=> Modelo", "SIGNIFICATIVO" if modelo.f_pvalue < alpha else "NÃO significativo")
print("\nTeste t por coeficiente:")
for nome in modelo.params.index:
    p = modelo.pvalues[nome]
    print(f"  {nome:<12} t = {modelo.tvalues[nome]:8.3f} | p = {p:.4e} | "
          f"{'significativo' if p < alpha else 'não significativo'}")
print("\nIntervalos de confiança (95%):")
print(modelo.conf_int().rename(columns={0: "2.5%", 1: "97.5%"}).round(4).to_string())

# ------------------------------------------------------------------
# 5. IMPORTÂNCIA RELATIVA (coeficientes padronizados)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("5) COEFICIENTES PADRONIZADOS (beta)")
print("=" * 70)
beta_pad = modelo.params[XS] * df[XS].std() / df[Y].std()
print(beta_pad.sort_values(ascending=False).round(4).to_string())
print(f"\nVariável mais importante: {beta_pad.abs().idxmax()}")

# Conferência: regressão com variáveis padronizadas (z-score)
z = (df - df.mean()) / df.std()
mz = sm.OLS(z[Y], z[XS]).fit()
print("\nConferência via z-score:", mz.params.round(4).to_dict())

# Multicolinearidade (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
    index=XS,
)
print("\nVIF:")
print(vif.round(3).to_string())

# ------------------------------------------------------------------
# 6. PREVISÃO E AVALIAÇÃO
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("6) PREVISÃO E AVALIAÇÃO")
print("=" * 70)

# >>> Altere os valores do novo cenário aqui <<<
novo = pd.DataFrame({"Registros": [80], "Variaveis": [15], "Estimadores": [150]})
pred = modelo.get_prediction(sm.add_constant(novo, has_constant="add")).summary_frame(alpha=0.05)
print("Novo cenário:", novo.iloc[0].to_dict())
print(f"Tempo previsto = {pred['mean'].iloc[0]:.2f}")
print(f"IC 95% da média     = [{pred['mean_ci_lower'].iloc[0]:.2f}, {pred['mean_ci_upper'].iloc[0]:.2f}]")
print(f"IC 95% de predição  = [{pred['obs_ci_lower'].iloc[0]:.2f}, {pred['obs_ci_upper'].iloc[0]:.2f}]")

df["Previsto"] = modelo.fittedvalues
df["Residuo"] = modelo.resid

# Gráficos Real × Previsto e Resíduos
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

lim = [df[Y].min() - 3, df[Y].max() + 3]
axes[0].scatter(df[Y], df["Previsto"], s=45)
axes[0].plot(lim, lim, "r--")
axes[0].set(xlabel="Real", ylabel="Previsto", title="Real × Previsto")

axes[1].scatter(df["Previsto"], df["Residuo"], s=45)
axes[1].axhline(0, color="r", ls="--")
axes[1].set(xlabel="Previsto", ylabel="Resíduo", title="Resíduos × Previsto")

sm.qqplot(df["Residuo"], line="s", ax=axes[2])
axes[2].set_title("Q-Q plot dos resíduos")

plt.tight_layout()
plt.savefig("fig_diagnostico.png", dpi=150)

# Diagnóstico dos resíduos
sw = stats.shapiro(df["Residuo"])
print("\nDiagnóstico dos resíduos:")
print(f"  Média dos resíduos         = {df['Residuo'].mean():.2e}")
print(f"  Resíduo mín / máx          = {df['Residuo'].min():.3f} / {df['Residuo'].max():.3f}")
print(f"  Shapiro-Wilk (normalidade) : W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}")
print(f"  Durbin-Watson              = {durbin_watson(df['Residuo']):.4f}")

# Métricas de erro
mae = df["Residuo"].abs().mean()
rmse = np.sqrt((df["Residuo"] ** 2).mean())
mape = (df["Residuo"].abs() / df[Y]).mean() * 100
print(f"  MAE = {mae:.3f} | RMSE = {rmse:.3f} | MAPE = {mape:.2f}%")

plt.show()
