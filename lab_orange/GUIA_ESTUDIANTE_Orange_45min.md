# Mini-lab Orange (45 min): churn, CLV y la importancia de elegir bien los parámetros

**Idea de la clase:** en Marketing Analytics programamos todo en Python. Hoy hacemos lo mismo arrastrando cajitas (Orange) y descubrimos dos cosas:
1. Arrastrar y soltar es rapidísimo para probar ideas.
2. Un modelo con los parámetros por defecto puede verse "bien" y ser malo. Elegirlos es parte del trabajo.

**Datos:** `clientes_churn.csv` (sintético): 6,000 clientes, 16 columnas, `churn_90d` = ¿se fue en los siguientes 90 días? (≈23% sí). Tiene 6 columnas `ruido_1..6` que no sirven para nada (a propósito).

---
## Cargar y preparar (drag & drop)
1. Widget **File** → abrir `clientes_churn.csv`.
2. Doble clic en las columnas y cambiar el *Role*:
   - `churn_90d` → **target**
   - `customer_id` → **meta** (un ID no es una característica; si se queda como feature el modelo se confunde)
   - `clv_estimado` → **meta** (lo usamos después para el dinero, no para predecir)

## Cuatro modelos con parámetros POR DEFECTO
Conectar File → **Tree**, **kNN**, **Logistic Regression**, **Random Forest** → todos a **Test and Score** (Cross validation, 10 folds, target = si). Agregar **ROC Analysis** y **Confusion Matrix**.

Anoten AUC y CA (exactitud) de cada uno. Referencia (±0.02): Tree ≈ 0.56 · kNN ≈ 0.60 · Random Forest ≈ 0.70 · Logistic ≈ 0.77.
Preguntas:
- Si siempre diriamos "no" (es decir todos se quedan) tiene CA ≈ 0.77. ¿Cuáles modelos lo superan en CA? ¿Y en AUC? **¿Por qué AUC es mejor aquí?**
- El árbol es el más "inteligente" en apariencia. ¿Cómo le fue? Esa es la pista del siguiente bloque.

## Mover parámetros a mano (arrastrar)
1. Copiar el widget **Tree** dos veces (copy/paste) y dejar los tres conectados a Test and Score.
2. Tree #2: *Limit depth* = **4**. Tree #3: *Limit depth* = **2**. (Tree #1 queda en 100.)
3. Cambien también kNN: *Neighbors* 5 → **100**.
4. Test and Score se actualiza solo. Referencia: Tree prof. 100 ≈ 0.56 · prof. 4 ≈ 0.71 · prof. 2 ≈ 0.68 · kNN k=100 ≈ 0.65.
5. Agreguen un widget **Tree Viewer** al de profundidad 100 y al de 4. ¿Cuál se puede explicar en una reunión con Marketing?

**Idea clave:** el árbol profundo "memoriza" a los clientes (se aprende hasta el ruido). Lo vamos a ver con números en el siguiente bloque.

## Experimentos automáticos con código (Python Script)
Conectar File → **Python Script**. Pegar el contenido de `scripts/script1_barrido_parametros.py` y presionar *Run*.
Prueba 20+ configuraciones en segundos y dibuja 3 gráficas. Observen:
- **Árbol:** el AUC de *entrenamiento* sube hasta 0.98 mientras el de *validación* baja de 0.70 a 0.57. Mejor profundidad ≈ **4**.
- **kNN:** k=1 ≈ 0.56 (muy nervioso) → mejor cerca de k≈50–100 (≈ 0.65).
- **Logística:** casi no le importa C, salvo si es exageradamente pequeño (0.001 → 0.74). **No todos los parámetros importan igual.**

Reto (5 min): editen `profundidades` para probar también `min_samples_leaf` = 20, 50. ¿Mejora el árbol?

## De la predicción al dinero (CLV)
Python Script nuevo con `scripts/script2_umbral_ganancia.py`.
Supuesto: contactar cuesta US$12; de los que se iban y contactamos, 25% se queda y nos deja su `clv_estimado`.
Referencia: Logística → contactar ≈ **22%** de la base, ganancia ≈ US$7,200 · Árbol prof. 4 → ≈ US$4,400 · Árbol por defecto → ≈ US$1,900.
- Cambien `COSTO_CONTACTO` a 3 y a 30. ¿Cambia el % óptimo? **El corte también es un parámetro, y depende del negocio.**
- Mismo cliente, mismo dato, ¿cuánto dinero perdió quien dejó el árbol por defecto?

## Para entregar (opcional, 1 página)
Captura del flujo de Orange + tabla de 4 filas (modelo, parámetro elegido, AUC) + 3 líneas: ¿qué parámetro importó más y cómo lo supieron?
