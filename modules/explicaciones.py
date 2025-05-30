import streamlit as st

def mostrar_explicacion_regresion_lineal():
    st.title("📈 Explicación paso a paso de Regresión Lineal Simple")

    st.markdown(r"""
    ### ¿Qué es la Regresión Lineal Simple?

    Es un método estadístico para modelar la relación entre una variable dependiente \(Y\) y una variable independiente \(X\) usando una línea recta.

    La fórmula general de la recta es:
    """)
    st.latex(r"Y = \beta_0 + \beta_1 X + \varepsilon")

    st.markdown(r"""
    Donde:  
    - \(\beta_0\) es el intercepto (ordenada al origen).  
    - \(\beta_1\) es la pendiente (cambio esperado en \(Y\) por unidad de cambio en \(X\)).  
    - \(\varepsilon\) es el término de error o residual.

    ---

    ### Objetivo

    Encontrar los valores de \(\beta_0\) y \(\beta_1\) que minimicen el error cuadrático entre los valores observados y los predichos por el modelo.

    ---

    ### Paso 1: Cálculo de medias

    Se calcula la media de \(X\) y de \(Y\):

    \[
    \bar{X} = \frac{1}{n} \sum_{i=1}^n X_i \quad,\quad \bar{Y} = \frac{1}{n} \sum_{i=1}^n Y_i
    \]

    ---

    ### Paso 2: Cálculo de la pendiente \(\beta_1\)

    \[
    \beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2}
    \]

    También expresado como:

    \[
    \beta_1 = \frac{n \sum X_i Y_i - \sum X_i \sum Y_i}{n \sum X_i^2 - (\sum X_i)^2}
    \]

    ---

    ### Paso 3: Cálculo del intercepto \(\beta_0\)

    \[
    \beta_0 = \bar{Y} - \beta_1 \bar{X}
    \]

    ---

    ### Paso 4: Ecuación final del modelo

    \[
    \hat{Y} = \beta_0 + \beta_1 X
    \]

    donde \(\hat{Y}\) es el valor predicho.

    ---

    ### Paso 5: Evaluación del modelo

    Se utilizan métricas como:

    - **Error Cuadrático Medio (MSE):**

    \[
    MSE = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat{Y}_i)^2
    \]

    - **Coeficiente de Determinación \(R^2\):**

    \[
    R^2 = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2}
    \]

    Que indica qué proporción de la variabilidad de \(Y\) es explicada por \(X\).

    ---

    ### Interpretación

    - Si \(\beta_1 > 0\), \(Y\) tiende a aumentar cuando \(X\) aumenta.  
    - Si \(\beta_1 < 0\), \(Y\) tiende a disminuir cuando \(X\) aumenta.  
    - Si \(R^2\) está cerca de 1, el modelo explica bien la relación.  
    - Si está cerca de 0, el modelo explica poco.

    ---

    ### Uso práctico

    Una vez calculados \(\beta_0\) y \(\beta_1\), puedes predecir \(Y\) para cualquier nuevo valor de \(X\) usando la fórmula del modelo.

    """)


def mostrar_explicacion_regresion_multiple():
    st.title("📊 Explicación paso a paso de Regresión Lineal Múltiple")

    st.markdown(r"""
    ### ¿Qué es la Regresión Lineal Múltiple?

    Es una extensión de la regresión lineal simple que modela la relación entre una variable dependiente \(Y\) y múltiples variables independientes \(X_1, X_2, ..., X_n\).

    La fórmula general es:

    \[
    Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n + \varepsilon
    \]

    Donde:  
    - \(\beta_0\) es el intercepto.  
    - \(\beta_1, \beta_2, ..., \beta_n\) son los coeficientes de las variables independientes.  
    - \(\varepsilon\) es el término de error.

    ---

    ### Objetivo

    Encontrar los coeficientes \(\beta\) que minimizan el error cuadrático entre los valores observados y los predichos, usando la suma de residuos al cuadrado.

    ---

    ### Representación matricial

    El modelo puede representarse como:

    \[
    \mathbf{Y} = \mathbf{X} \boldsymbol{\beta}
    \]

    Donde:

    \[
    \mathbf{Y} =
    \begin{bmatrix}
    Y_1 \\ Y_2 \\ \vdots \\ Y_m
    \end{bmatrix}
    ,\quad
    \mathbf{X} =
    \begin{bmatrix}
    1 & X_{11} & X_{12} & \cdots & X_{1n} \\
    1 & X_{21} & X_{22} & \cdots & X_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & X_{m1} & X_{m2} & \cdots & X_{mn}
    \end{bmatrix}
    ,\quad
    \boldsymbol{\beta} =
    \begin{bmatrix}
    \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_n
    \end{bmatrix}
    \]

    ---

    ### Cálculo de los coeficientes

    Usamos la ecuación normal:

    \[
    \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
    \]

    Esta fórmula proporciona los coeficientes que minimizan la suma de errores cuadrados.

    ---

    ### Interpretación de coeficientes

    Cada \(\beta_i\) representa el cambio esperado en \(Y\) por unidad de cambio en \(X_i\), manteniendo las demás variables constantes.

    ---

    ### Ejemplo práctico

    Supongamos que queremos predecir el precio de una casa según su tamaño (m²) y número de habitaciones:

    \[
    Precio = 31.04 + 1.4 \times Tamaño + 2.5 \times Habitaciones
    \]

    Esto indica que por cada metro cuadrado adicional, el precio aumenta en 1.4 unidades monetarias, y por cada habitación adicional, aumenta en 2.5 unidades.

    ---

    ### Uso práctico

    La ecuación resultante puede usarse para hacer predicciones basadas en múltiples variables independientes.

    """)

def mostrar_explicacion_k_means():
    st.title("📌 Explicación paso a paso del algoritmo K-means")

    st.markdown(r"""
    K-means es un algoritmo de clustering que agrupa datos en \(k\) clusters basados en la distancia a centroides.

    ### Proceso general:

    1. Se elige el número de clusters \(k\).
    2. Se inicializan los centroides (aleatoriamente o por métodos heurísticos).
    3. Cada punto se asigna al cluster cuyo centroide está más cercano (usualmente distancia euclidiana).
    4. Se recalculan los centroides como la media de los puntos asignados a cada cluster.
    5. Se repiten los pasos 3 y 4 hasta que las asignaciones no cambien (convergencia).
    
    ### Características:

    - Puede trabajar con datos multidimensionales.
    - Es sensible a la inicialización de centroides.
    - No garantiza un óptimo global, pero suele converger rápido.
    - Es común usar reducción dimensional para visualizar clusters cuando hay muchas variables.

    ### Aplicaciones:

    - Segmentación de clientes.
    - Agrupamiento de documentos.
    - Detección de patrones en datos.
    """)
def mostrar_explicacion_k_modas():
    st.title("Explicación de K-modas")
    st.markdown("""
    ## ¿Qué es K-modas?

    K-modas es una técnica de clustering para datos categóricos. Similar a K-means, pero en lugar de usar medias y distancia euclidiana, usa modas (valores más frecuentes) y una medida de disimilitud basada en conteo de diferencias.

    ### Proceso básico:
    1. Se asignan los modos iniciales por cluster según la moda en los datos conocidos.
    2. Cada punto se asigna al cluster cuyo modo tiene menor número de diferencias con el dato.
    3. Se recalculan los modos con las asignaciones actuales.
    4. Se repite hasta convergencia.
    5. Se imputan los valores faltantes usando el modo del cluster asignado.

    ### Aplicaciones comunes:
    - Datos categóricos puros.
    - Segmentación de clientes por atributos categóricos.
    - Análisis de comportamiento con variables no numéricas.

    ### Diferencias con K-means:
    - K-means usa medias y distancia euclidiana (numérico).
    - K-modas usa modas y distancia por conteo de diferencias (categórico).

    """)
def mostrar_explicacion_id3():
    st.title("🌟 Explicación del Árbol de Decisión ID3")
    st.markdown(r"""
**Proceso de construcción del Árbol de Decisión ID3**

1. **Introducción**  
El algoritmo ID3 construye un árbol de decisión que clasifica datos buscando, en cada nodo, el atributo que maximiza la ganancia de información (reducción de entropía).

2. **Entropía**  
Mide la impureza o incertidumbre de un conjunto de datos:
""")
    st.latex(r"E(S) = - \sum_{i=1}^{c} p_i \log_k(p_i)")
    st.markdown(r"""
- \(S\): conjunto de datos.  
- \(c\): número de clases.  
- \(p_i\): proporción de ejemplos en la clase \(i\).

Si todas las instancias son de la misma clase, \(E(S)=0\). Si están uniformemente distribuidas, \(E(S)\) es máximo.

3. **Ganancia de Información**  
Cuantifica cuánto disminuye la entropía al dividir por un atributo \(A\):
""")
    st.latex(r"G(S,A) = E(S) - \sum_{v \in Valores(A)} \frac{|S_v|}{|S|}\,E(S_v)")
    st.markdown(r"""
- \(S_v\): subconjunto de \(S\) donde el atributo \(A\) toma el valor \(v\).

Se elige el atributo con **mayor** \(G(S,A)\) (o **menor** entropía condicional).

4. **Proceso recursivo**  
- Calcular \(E(S)\).  
- Para cada atributo restante, calcular \(G(S,A)\).  
- Crear un nodo con el mejor atributo y dividir \(S\) según sus valores.  
- Repetir en cada rama hasta que:  
  1. Todas las instancias queden en la misma clase (entropía = 0).  
  2. No queden atributos.

5. **Construcción del árbol**  
- **Nodos internos**: representan pruebas sobre atributos.  
- **Ramas**: representan los valores de esos atributos.  
- **Hojas**: contienen las clases finales.

6. **Extracción de reglas**  
Cada camino desde la raíz hasta una hoja produce una regla del tipo:  
```
Si A1 = v1 y A2 = v2 …, entonces Clase = c
```"""
)
