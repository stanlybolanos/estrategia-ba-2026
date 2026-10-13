# ---- Python Script widget: BARRIDO AUTOMATICO DE PARAMETROS ----
# Conexion: File -> Python Script.  (en el File widget: churn_90d = target; customer_id y clv_estimado = meta)
import numpy as np
import matplotlib.pyplot as plt
from Orange.classification import TreeLearner, KNNLearner, LogisticRegressionLearner
from Orange.evaluation import CrossValidation, TestOnTrainingData, AUC

data = in_data
def auc_cv(learner):      # AUC en datos que el modelo NO vio (validacion cruzada, 5 pliegues)
    return AUC(CrossValidation(k=5, random_state=0)(data, [learner]))[0]
def auc_train(learner):   # AUC en los mismos datos con los que se entreno
    return AUC(TestOnTrainingData()(data, [learner]))[0]

# --- Experimento 1: profundidad maxima del arbol -------------------------------
profundidades = [1, 2, 3, 4, 5, 6, 8, 10, 15, 30]
tr, cv = [], []
for d in profundidades:
    L = TreeLearner(max_depth=d, min_samples_leaf=2, min_samples_split=5)
    tr.append(auc_train(L)); cv.append(auc_cv(L))
    print(f"max_depth={d:>3}  AUC entrenamiento={tr[-1]:.3f}  AUC validacion={cv[-1]:.3f}")
mejor = profundidades[int(np.argmax(cv))]
print(f"\n>>> Mejor profundidad: {mejor}  (AUC validacion {max(cv):.3f})")
print(f">>> Con el valor por defecto (profundidad 100) el AUC de validacion cae a ~{cv[-1]:.3f}\n")

# --- Experimento 2: k vecinos en kNN --------------------------------------------
ks = [1, 3, 5, 10, 20, 50, 100, 200]
kc = []
for k in ks:
    kc.append(auc_cv(KNNLearner(n_neighbors=k)))
    print(f"k={k:>3}  AUC validacion={kc[-1]:.3f}")
print(f"\n>>> Mejor k: {ks[int(np.argmax(kc))]}")

# --- Experimento 3: regularizacion C en regresion logistica -----------------------
Cs = [0.001, 0.01, 0.1, 1, 10]
lc = [auc_cv(LogisticRegressionLearner(C=c)) for c in Cs]
for c, a in zip(Cs, lc): print(f"C={c:<6} AUC validacion={a:.3f}")

# --- Graficas --------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(profundidades, tr, 'o--', label='entrenamiento'); ax[0].plot(profundidades, cv, 'o-', label='validacion')
ax[0].set_title('Arbol: profundidad'); ax[0].set_xlabel('max_depth'); ax[0].set_ylabel('AUC'); ax[0].legend()
ax[1].plot(ks, kc, 'o-'); ax[1].set_xscale('log'); ax[1].set_title('kNN: numero de vecinos'); ax[1].set_xlabel('k')
ax[2].plot(Cs, lc, 'o-'); ax[2].set_xscale('log'); ax[2].set_title('Logistica: C'); ax[2].set_xlabel('C')
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.show()
