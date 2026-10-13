# ---- Python Script widget: ¿A CUANTOS CLIENTES CONTACTAR? (conecta churn con CLV) ----
# Conexion: File -> Python Script.  Supuestos de negocio (cambien y vean que pasa):
COSTO_CONTACTO = 12.0    # US$ por cliente contactado (cupon + envio)
TASA_SALVADO   = 0.25    # de los que se iban y contactamos, 25% se queda
import numpy as np
import matplotlib.pyplot as plt
from Orange.classification import TreeLearner, LogisticRegressionLearner
from Orange.evaluation import CrossValidation

data = in_data
clv = data.metas[:, [v.name for v in data.domain.metas].index('clv_estimado')].astype(float)
y = (data.Y == list(data.domain.class_var.values).index('si')).astype(float)
n = len(y)

def curva(learner):
    res = CrossValidation(k=5, random_state=0)(data, [learner])
    p = res.probabilities[0][:, list(data.domain.class_var.values).index('si')]
    orden = res.row_indices[np.argsort(-p)]            # de mayor a menor riesgo
    ganancia = np.cumsum(y[orden] * TASA_SALVADO * clv[orden] - COSTO_CONTACTO)
    return ganancia

fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(1, n + 1) / n * 100
for nombre, L in [('Logistica', LogisticRegressionLearner(C=1)),
                  ('Arbol profundidad 4', TreeLearner(max_depth=4, min_samples_leaf=2, min_samples_split=5)),
                  ('Arbol por defecto (100)', TreeLearner(max_depth=100, min_samples_leaf=2, min_samples_split=5))]:
    g = curva(L); i = int(np.argmax(g))
    print(f"{nombre:<26} mejor corte: contactar {x[i]:.0f}% de la base -> ganancia US$ {g[i]:,.0f}")
    ax.plot(x, g, label=nombre)
ax.axhline(0, color='k', lw=.8); ax.set_xlabel('% de la base contactada (de mayor a menor riesgo)')
ax.set_ylabel('Ganancia acumulada (US$)'); ax.legend(); ax.grid(alpha=.3); ax.set_title('Elegir el corte tambien es un parametro')
plt.tight_layout(); plt.show()
