import streamlit as st










def mostrar_explicacion_id3():
    st.title("Explicación del algoritmo Árbol de Decisión ID3")

    st.markdown(r"""
    # Proceso de construcción del Árbol de Decisión ID3

    1. **Introducción**  
    El algoritmo ID3 construye un árbol de decisión que clasifica datos usando el criterio de máxima ganancia de información, basada en la entropía.

    2. **Entropía**  
    Mide la impureza o incertidumbre de un conjunto de datos.  
    Fórmula:
    """)
    st.latex("Entropía(S) = - \\sum_{i=1}^c p_i \\log_2(p_i)")
    st.markdown(r"""
    donde:  
    - \(S\) es el conjunto de datos,  
    - \(c\) es el número de clases,  
    - \(p_i\) es la proporción de ejemplos en la clase \(i\).

    Si todos los datos pertenecen a una sola clase, la entropía es 0 (conjunto puro).  
    Si las clases están distribuidas uniformemente, la entropía es máxima.

    3. **Ganancia de Información**  
    Mide cuánto reduce la entropía un atributo al dividir los datos.  
    Fórmula:
    """)
    st.latex("Ganancia(S, A) = Entropía(S) - \\sum_{v \\in Valores(A)} \\frac{|S_v|}{|S|} \\cdot Entropía(S_v)")
    st.markdown(r"""
    donde:  
    - \(S_v\) es el subconjunto de \(S\) donde el atributo \(A\) toma el valor \(v\).

    Elegimos el atributo con **máxima ganancia** para dividir.

    4. **Proceso Recursivo**  
    Calcula la entropía y ganancia para cada atributo.  
    Escoge el atributo con mayor ganancia para crear un nodo.  
    Divide el conjunto según los valores del atributo.  
    Repite recursivamente en cada subconjunto hasta que:  
    - Todos los ejemplos son de la misma clase (entropía = 0).  
    - No quedan más atributos para dividir.

    5. **Construcción del Árbol**  
    El nodo raíz es el atributo con mayor ganancia.  
    Cada rama corresponde a un valor del atributo.  
    Las hojas contienen las clases finales.

    6. **Extracción de Reglas**  
    Cada camino desde la raíz hasta una hoja representa una regla.  
    La regla concatena las condiciones de cada nodo en el camino.  

    Ejemplo:  
    `Si Nivel académico = Magíster y Estrato socioeconómico = Medio y Área de estudio = Ingeniería, entonces Categoría = Titular`
    """)


def mostrar_explicacion_regresion_lineal():
    st.title("📈 Explicación paso a paso de Regresión Lineal Simple")

    st.markdown("""
    ### ¿Qué es la Regresión Lineal Simple?

    Es un método estadístico para modelar la relación entre una variable dependiente \(Y\) y una variable independiente \(X\) usando una línea recta.

    La fórmula general de la recta es:
    """)
    st.latex(r"Y = \beta_0 + \beta_1 X + \varepsilon")

    st.markdown("""
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


def mostrar_explicacion_k_means():
    st.title("Explicación K-means")
    st.markdown("""
    K-means es un algoritmo de clustering que particiona los datos en K grupos basados en la distancia entre puntos y centroides.
    ...
    """)
    # Contenido detallado de la explicación K-means