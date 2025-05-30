import streamlit as st

# -----------------------------
# Explicación: Regresión Lineal Simple
# -----------------------------

def mostrar_explicacion_regresion_lineal():
    st.title("📈 Explicación paso a paso de Regresión Lineal Simple")

    st.markdown(r"""
**1. ¿Qué es la Regresión Lineal Simple?**  
Es un método estadístico que modela la relación entre:
- Variable dependiente \(Y\)
- Variable independiente \(X\)
mediante una línea recta.
""")
    st.latex(r"Y = \beta_0 + \beta_1 X + \varepsilon")

    st.markdown(r"""
**2. Objetivo**  
Encontrar \(\beta_0\) y \(\beta_1\) que minimicen el **error cuadrático** entre los valores observados \(Y_i\) y los predichos \(\hat{Y}_i\).
""")

    with st.expander("🔢 Paso 1: Cálculo de medias", expanded=True):
        st.markdown(r"""Se calcula la media de \(X\) y de \(Y\):""")
        st.latex(r"\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i \quad;\quad \bar{Y} = \frac{1}{n} \sum_{i=1}^n Y_i")

    with st.expander("🔢 Paso 2: Cálculo de la pendiente \(\beta_1\)", expanded=False):
        st.markdown(r"Dos fórmulas equivalentes:")
        st.latex(r"\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2}")
        st.latex(r"\beta_1 = \frac{n \sum X_i Y_i - \sum X_i \sum Y_i}{n \sum X_i^2 - (\sum X_i)^2}")

    with st.expander("🔢 Paso 3: Cálculo del intercepto \(\beta_0\)", expanded=False):
        st.markdown(r"\(\beta_0 = \bar{Y} - \beta_1 \bar{X}\)")

    st.markdown(r"""
**4. Ecuación final del modelo**  
\[
\hat{Y} = \beta_0 + \beta_1 X
\]
""")

    with st.expander("📊 Paso 5: Evaluación del modelo", expanded=False):
        st.markdown("**Métricas comunes:**")
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat{Y}_i)^2")
        st.latex(r"R^2 = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2}")

    st.markdown(r"""
**6. Interpretación**  
- \(\beta_1 > 0\): \(Y\) aumenta con \(X\).  
- \(\beta_1 < 0\): \(Y\) disminuye con \(X\).  
- \(R^2\) cercano a 1 indica buen ajuste.
""")

    with st.expander("🔢 Paso 7: Uso práctico", expanded=False):
        st.markdown(r"""
```python
# Ejemplo de predicción
X_nuevo = [1, valor_X]
pred = X_nuevo @ [beta_0, beta_1]
```
""")

# -----------------------------
# Explicación: Regresión Lineal Múltiple
# -----------------------------

def mostrar_explicacion_regresion_multiple():
    st.title("📊 Explicación paso a paso de Regresión Lineal Múltiple")

    st.markdown(r"""
**1. ¿Qué es la Regresión Lineal Múltiple?**  
Extiende la regresión lineal simple a varias variables independientes \(X_1, \,..., X_p\) para predecir \(Y\).

Ecuación general:  
\[
Y = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p + \varepsilon
\]

- \(\beta_0\): intercepto.  
- \(\beta_j\): coeficientes de \(X_j\).  
- \(\varepsilon\): término de error.
""")

    st.markdown(r"""
**2. Objetivo**  
Minimizar la suma de cuadrados de residuos (RSS):
\[
RSS = \sum_{i=1}^n (Y_i - (\beta_0 + \sum_{j=1}^p \beta_j X_{ij}))^2
\]
""")

    with st.expander("🔢 Paso 3: Ecuación Normal", expanded=True):
        st.markdown("Fórmula de MCO (ecuación normal):")
        st.latex(r"\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T Y")

    with st.expander("🔢 Paso 4: Coeficientes", expanded=False):
        st.markdown(r"Se extraen los coeficientes del vector \(\hat{\boldsymbol{\beta}}\):")
        st.latex(r"\hat{\beta}_j = [(X^T X)^{-1} X^T Y]_j")

    with st.expander("📊 Paso 5: Evaluación", expanded=False):
        st.markdown("**Métricas:**")
        st.latex(r"MSE = \frac{1}{n} \sum (Y_i - \hat{Y}_i)^2")
        st.latex(r"R^2 = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2}")

    st.markdown(r"""
**6. Interpretación**  
- \(\beta_j\): cambio en \(Y\) por unidad de \(X_j\).  
- \(R^2\): cercano a 1 indica buen ajuste.
""")

    with st.expander("🔢 Paso 7: Uso práctico", expanded=False):
        st.markdown(r"""
```python
# Ejemplo de predicción
X_nuevo = [1, x1, ..., xp]
pred = X_nuevo @ beta_hat
```
""")

# -----------------------------
# Explicación: K-means
# -----------------------------

def mostrar_explicacion_k_means():
    st.title("📌 Explicación paso a paso de K-means")
    st.markdown(r"""
**Proceso general**:  
1. Elegir \(k\).  
2. Inicializar centroides.  
3. Asignar cada punto al centroide más cercano.  
4. Recalcular centroides como medias.  
5. Repetir hasta convergencia.

**Características**: rápido, sensible a inicialización, no garantiza óptimo global.

**Aplicaciones**: segmentación de clientes, clustering de documentos.
""")

# -----------------------------
# Explicación: K-modes
# -----------------------------

def mostrar_explicacion_k_modas():
    st.title("📌 Explicación paso a paso de K-modes (K-modas)")
    st.markdown(r"""
**Proceso básico**:  
1. Inicializar modas.  
2. Asignar puntos al modo más similar.  
3. Recalcular modas.  
4. Repetir hasta convergencia.

Usa conteo de diferencias para datos categóricos; imputa faltantes con modas.
""")

# -----------------------------
# Explicación: Árbol de Decisión ID3
# -----------------------------

def mostrar_explicacion_id3():
    st.title("🌟 Explicación paso a paso de Árbol de Decisión ID3")
    st.markdown(r"""
**1. Entropía**  
\(E(S) = -\sum_i p_i \log_k(p_i)\)

**2. Ganancia de Información**  
\(G(S,A) = E(S) - \sum_v \frac{|S_v|}{|S|} E(S_v)\)

**3. Selección**  
Elegir atributo con mayor \(G(S,A)\) (o menor \(E(S|A)\)).

**4. Recursión**  
Repetir hasta nodos puros o sin atributos.

**5. Reglas**  
Cada camino raíz→hoja define una regla.
""")

# Ejemplo de uso:
# if st.sidebar.button("Mostrar explicaciones"):
#     mostrar_explicacion_regresion_lineal()
#     mostrar_explicacion_regresion_multiple()
#     mostrar_explicacion_k_means()
#     mostrar_explicacion_k_modas()
#     mostrar_explicacion_id3()
