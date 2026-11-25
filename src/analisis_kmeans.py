"""
Proyecto integrador: Segmentación de clientes del departamento de perfumería
de una tienda departamental mediante K-Means

Autora: Viviana Anahí Olmedo García

Dataset propio (simulado): data/original/clientes_perfumeria.csv, generado con
src/generar_dataset.py. Ejecutar desde la carpeta raíz del proyecto:

    python src/analisis_kmeans.py

El script realiza las partes 2 a 5 del proyecto:
    Parte 2. Obtención y exploración del dataset
    Parte 3. Limpieza y preparación de datos
    Parte 4. Implementación del algoritmo K-Means
    Parte 5. Análisis e interpretación de resultados

Genera:
    data/procesado/clientes_perfumeria_clusters.csv  (dataset con clúster asignado)
    graficas/*.png                                   (gráficas del análisis)
    resultados/*.csv, resultados/resumen.json        (tablas y métricas)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las gráficas como imagen sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_ORIGINAL = RAIZ / "data" / "original" / "clientes_perfumeria.csv"
RUTA_PROCESADO = RAIZ / "data" / "procesado" / "clientes_perfumeria_clusters.csv"
DIR_GRAFICAS = RAIZ / "graficas"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_GRAFICAS.mkdir(exist_ok=True)
DIR_RESULTADOS.mkdir(exist_ok=True)
RUTA_PROCESADO.parent.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# Paleta pastel (validada para daltonismo); un color fijo por segmento, igual que el dashboard
MORADO, ROSA, DORADO, VERDE, GRIS = "#8C84D8", "#E07FA6", "#D9A447", "#4DAE90", "#7A7383"
DIVERGENTE = LinearSegmentedColormap.from_list("rosa_lavanda", ["#D4668F", "#F4F1EE", "#6E63C2"])
sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams.update({
    "font.family": ["Segoe UI", "DM Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": "#2B2733", "axes.labelcolor": "#5E5867", "axes.edgecolor": "#E2DCD5",
    "xtick.color": "#5E5867", "ytick.color": "#5E5867", "grid.color": "#EFEAE4",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.titleweight": "semibold", "figure.dpi": 110, "legend.frameon": False})


def titulo(texto):
    print("\n" + "=" * 72)
    print(texto)
    print("=" * 72)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(DIR_GRAFICAS / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> gráfica guardada: graficas/{nombre}")


NUMERICAS = ["edad", "antiguedad_meses", "compras_anuales", "gasto_anual",
             "ticket_promedio", "dias_ultima_compra", "pct_compras_descuento",
             "satisfaccion"]
ETIQUETA = {
    "edad": "Edad (años)", "antiguedad_meses": "Antigüedad (meses)",
    "compras_anuales": "Compras al año", "gasto_anual": "Gasto anual (MXN)",
    "ticket_promedio": "Ticket promedio (MXN)",
    "dias_ultima_compra": "Días desde la última compra",
    "pct_compras_descuento": "% de compras con descuento",
    "satisfaccion": "Satisfacción (1-5)",
}

# ==================================================================
# PARTE 2. OBTENCIÓN Y EXPLORACIÓN DEL DATASET
# ==================================================================
titulo("PARTE 2. OBTENCIÓN Y EXPLORACIÓN DEL DATASET")

df_original = pd.read_csv(RUTA_ORIGINAL)

print("\nPrimeros 10 registros:")
print(df_original.head(10))

print(f"\nDimensiones: {df_original.shape[0]} registros x {df_original.shape[1]} columnas")

print("\nInformación general del DataFrame (columnas, tipos y no nulos):")
df_original.info()
print("  Nota: 'gasto_anual' aparece como texto (object/str) porque algunas filas")
print("  traen el monto con '$' y comas. Se corrige en la Parte 3.")

print("\nEstadísticas descriptivas (variables numéricas):")
print(df_original.describe().round(2).T)

print("\nEstadísticas de las variables categóricas:")
print(df_original.select_dtypes(exclude="number").describe().T)

print("\nValores nulos por columna (isnull().sum()):")
print(df_original.isnull().sum().to_string())
nulos = int(df_original.isnull().sum().sum())
duplicados = int(df_original.duplicated().sum())
print(f"\nValores nulos en todo el dataset: {nulos}")
print(f"Registros duplicados: {duplicados}")

print("\nValores distintos escritos en 'genero' (inconsistencias de captura):")
print(df_original["genero"].value_counts(dropna=False).to_string())

# Copia numérica rápida solo para las gráficas exploratorias
exp = df_original.drop_duplicates().copy()
exp["gasto_anual"] = pd.to_numeric(exp["gasto_anual"].astype(str)
                                   .str.replace(r"[$,]", "", regex=True))
exp.loc[~exp["edad"].between(18, 90), "edad"] = np.nan
exp["dias_ultima_compra"] = exp["dias_ultima_compra"].abs()

# Gráfica exploratoria 1: histogramas de las variables numéricas
fig, ejes = plt.subplots(2, 4, figsize=(17, 7))
for eje, var in zip(ejes.flat, NUMERICAS):
    sns.histplot(exp[var].dropna(), bins=30, ax=eje, color=MORADO, alpha=0.75,
                 discrete=(var == "satisfaccion"))
    eje.set_title(ETIQUETA[var])
    eje.set_xlabel("")
    eje.set_ylabel("Clientes" if var in ("edad", "ticket_promedio") else "")
fig.suptitle("Distribución de las variables numéricas de los clientes",
             fontweight="bold")
guardar(fig, "01_histogramas.png")

# Gráfica exploratoria 2: matriz de correlación
fig, eje = plt.subplots(figsize=(9, 7.5))
sns.heatmap(exp[NUMERICAS].corr(), annot=True, fmt=".2f", cmap=DIVERGENTE,
            vmin=-1, vmax=1, center=0, linewidths=1.5, linecolor="white", ax=eje,
            xticklabels=[ETIQUETA[v] for v in NUMERICAS],
            yticklabels=[ETIQUETA[v] for v in NUMERICAS],
            cbar_kws={"label": "Correlación de Pearson"})
eje.set_title("Matriz de correlación de las variables numéricas")
guardar(fig, "02_correlacion.png")

# Gráfica exploratoria 3: diagramas de caja estandarizados
z = (exp[NUMERICAS] - exp[NUMERICAS].mean()) / exp[NUMERICAS].std()
fig, eje = plt.subplots(figsize=(13, 5))
sns.boxplot(data=z.rename(columns=ETIQUETA), ax=eje, color="#D9D4F3", fliersize=3,
            flierprops={"markerfacecolor": ROSA, "markeredgecolor": ROSA})
eje.set_title("Diagramas de caja (valores estandarizados): detección de atípicos")
eje.set_ylabel("Desviaciones estándar respecto a la media")
eje.tick_params(axis="x", rotation=25)
guardar(fig, "03_boxplots.png")

# Gráfica exploratoria 4: variables categóricas
fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))
for eje, col, tit in [(ejes[0], "familia_olfativa", "Familia olfativa preferida"),
                      (ejes[1], "concentracion_preferida", "Concentración preferida"),
                      (ejes[2], "linea_preferida", "Línea de perfume preferida")]:
    conteo = exp[col].value_counts()
    eje.barh(conteo.index[::-1], conteo.values[::-1], color=MORADO, height=0.6)
    for i, v in enumerate(conteo.values[::-1]):
        eje.text(v + 5, i, str(v), va="center", fontsize=9)
    eje.set_xlim(0, conteo.max() * 1.15)
    eje.set_title(tit)
    eje.set_xlabel("Clientes")
fig.suptitle("Preferencias de compra de los clientes", fontweight="bold")
guardar(fig, "04_preferencias.png")

# Variables seleccionadas para el agrupamiento (modelo RFM + sensibilidad al precio)
VARS_MODELO = ["dias_ultima_compra", "compras_anuales", "gasto_anual",
               "pct_compras_descuento"]
print("\nVariables seleccionadas para el agrupamiento:", VARS_MODELO)
print("  - dias_ultima_compra (Recencia): qué tan reciente es el cliente")
print("  - compras_anuales (Frecuencia): cuántas veces compra al año")
print("  - gasto_anual (Monto): cuánto dinero deja en el departamento")
print("  - pct_compras_descuento: sensibilidad al precio / promociones")
print("  Son las tres variables del modelo RFM, estándar en segmentación de")
print("  clientes, más una variable de comportamiento ante promociones.")
print("  Se excluye ticket_promedio porque es gasto_anual / compras_anuales")
print("  (información repetida) y las variables demográficas se usan solo para")
print("  describir los grupos, no para formarlos.")

# ==================================================================
# PARTE 3. LIMPIEZA Y PREPARACIÓN DE DATOS
# ==================================================================
titulo("PARTE 3. LIMPIEZA Y PREPARACIÓN DE DATOS")

df = df_original.copy()
registros_inicio = len(df)
bitacora = []  # resumen de la limpieza para el documento y el dashboard

# 3.1 Registros duplicados
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n3.1 Duplicados eliminados: {registros_inicio - len(df)} "
      f"(registros: {registros_inicio} -> {len(df)})")
bitacora.append(["Registros duplicados", duplicados, "Se eliminaron con drop_duplicates()"])
ids_repetidos = int(df["id_cliente"].duplicated().sum())
print(f"    IDs de cliente repetidos después de eliminar duplicados: {ids_repetidos}")

# 3.2 Inconsistencias de texto
mapa_genero = {"f": "Femenino", "femenino": "Femenino", "mujer": "Femenino",
               "m": "Masculino", "masculino": "Masculino", "hombre": "Masculino"}
genero_antes = df["genero"].nunique()
df["genero"] = df["genero"].str.strip().str.lower().map(mapa_genero)
sucursal_antes = df["sucursal"].nunique()
df["sucursal"] = df["sucursal"].str.strip().str.title()
print("\n3.2 Inconsistencias de texto")
print(f"  genero: {genero_antes} formas distintas -> {df['genero'].nunique()} "
      f"({', '.join(df['genero'].dropna().unique())})")
print(f"  sucursal: {sucursal_antes} formas distintas -> {df['sucursal'].nunique()} "
      f"({', '.join(sorted(df['sucursal'].unique()))})")
bitacora.append(["Género escrito de distintas formas", f"{genero_antes} variantes",
                 "Se unificó en Femenino / Masculino"])
bitacora.append(["Sucursal con mayúsculas y espacios", f"{sucursal_antes} variantes",
                 "Se quitaron espacios y se unificó el formato"])

# 3.3 Tipos de datos
gasto_texto = int(df["gasto_anual"].astype(str).str.contains(r"\$").sum())
df["gasto_anual"] = pd.to_numeric(df["gasto_anual"].astype(str)
                                  .str.replace(r"[$,]", "", regex=True), errors="coerce")
print("\n3.3 Tipos de datos")
print(f"  gasto_anual: {gasto_texto} valores guardados como texto ('$12,345.67') "
      f"convertidos a número -> tipo {df['gasto_anual'].dtype}")
bitacora.append(["Gasto anual guardado como texto", f"{gasto_texto} valores",
                 "Se quitaron '$' y comas y se convirtió a float"])

# 3.4 Valores fuera de rango
edades_invalidas = int((~df["edad"].between(18, 90) & df["edad"].notna()).sum())
df.loc[~df["edad"].between(18, 90), "edad"] = np.nan
recencia_negativa = int((df["dias_ultima_compra"] < 0).sum())
df["dias_ultima_compra"] = df["dias_ultima_compra"].abs()
diferencia_ticket = (df["ticket_promedio"] - df["gasto_anual"] / df["compras_anuales"]).abs()
tickets_incoherentes = int((diferencia_ticket > 1).sum())
print("\n3.4 Valores fuera de rango")
print(f"  Edades imposibles (fuera de 18-90 años): {edades_invalidas} -> se "
      "marcan como nulas")
print(f"  Días desde la última compra negativos: {recencia_negativa} -> se toma "
      "el valor absoluto (error de signo al calcular la fecha)")
print(f"  Tickets que no cumplen ticket = gasto / compras: {tickets_incoherentes}")
bitacora.append(["Edades imposibles (0, 3, 150, 999...)", edades_invalidas,
                 "Se marcaron como nulas y se imputaron"])
bitacora.append(["Días desde la última compra negativos", recencia_negativa,
                 "Se corrigió el signo (valor absoluto)"])
bitacora.append(["Ticket diferente a gasto / compras", tickets_incoherentes,
                 "Ninguna (los datos son coherentes)"])

# 3.5 Valores nulos
print("\n3.5 Valores nulos antes del tratamiento:")
nulos_antes = df.isnull().sum()
print(nulos_antes[nulos_antes > 0].to_string())
mediana_edad = df["edad"].median()
mediana_satisf = df["satisfaccion"].median()
moda_canal = df["canal_preferido"].mode()[0]
df["edad"] = df["edad"].fillna(mediana_edad)
df["satisfaccion"] = df["satisfaccion"].fillna(mediana_satisf)
df["canal_preferido"] = df["canal_preferido"].fillna(moda_canal)
df["genero"] = df["genero"].fillna("No especificado")
df["familia_olfativa"] = df["familia_olfativa"].fillna("No especificada")
print(f"  edad -> mediana ({mediana_edad:.0f} años) | satisfaccion -> mediana "
      f"({mediana_satisf:.0f}) | canal_preferido -> moda ({moda_canal})")
print("  genero y familia_olfativa -> categoría 'No especificado/a' (inventar")
print("  un valor sesgaría los porcentajes por grupo)")
print(f"  Nulos restantes: {int(df.isnull().sum().sum())}")
bitacora.append(["Valores nulos", int(nulos_antes.sum()),
                 "Mediana (edad, satisfacción), moda (canal) y categoría "
                 "'No especificado' (género, familia)"])
df["edad"] = df["edad"].astype(int)
df["satisfaccion"] = df["satisfaccion"].astype(int)

# 3.6 Transformaciones y codificación
df["tarjeta_cod"] = df["tarjeta_departamental"].map({"Sí": 1, "No": 0})
df["canal_cod"] = df["canal_preferido"].map({"Tienda física": 0, "En línea": 1})
# Discretización de la edad con pd.cut (solo para describir y filtrar)
df["rango_edad"] = pd.cut(df["edad"], bins=[17, 25, 35, 50, 90],
                          labels=["18-25", "26-35", "36-50", "51 o más"])
print("\n3.6 Transformaciones")
print("  Codificación: tarjeta departamental Sí=1/No=0; canal Tienda=0/En línea=1")
print("  Rango de edad (pd.cut):")
print(df["rango_edad"].value_counts().sort_index().to_string())

# 3.7 Valores atípicos (regla 1.5 x IQR)
q1, q3 = df[VARS_MODELO].quantile(0.25), df[VARS_MODELO].quantile(0.75)
iqr = q3 - q1
atipicos = (df[VARS_MODELO] < q1 - 1.5 * iqr) | (df[VARS_MODELO] > q3 + 1.5 * iqr)
tabla_atipicos = pd.DataFrame({"atipicos_IQR": atipicos.sum(),
                               "limite_superior": (q3 + 1.5 * iqr).round(1)})
print("\n3.7 Valores atípicos por variable del modelo (regla 1.5 x IQR):")
print(tabla_atipicos)
registros_con_atipico = int(atipicos.any(axis=1).sum())
print(f"  Registros con al menos un atípico: {registros_con_atipico}")
print("  Los atípicos son clientes que gastan o compran mucho más que el resto.")
print("  Son clientes reales y valiosos, así que NO se eliminan. Tratamiento:")
print("   1) log(1 + x) en gasto_anual y compras_anuales (sesgo a la derecha)")
print("   2) recorte de las 4 variables a sus percentiles 1 y 99")

asimetria_antes = df[["gasto_anual", "compras_anuales"]].skew()
X_df = df[VARS_MODELO].copy()
X_df["gasto_anual"] = np.log1p(X_df["gasto_anual"])
X_df["compras_anuales"] = np.log1p(X_df["compras_anuales"])
asimetria_despues = X_df[["gasto_anual", "compras_anuales"]].skew()
p01, p99 = X_df.quantile(0.01), X_df.quantile(0.99)
X_recortado = X_df.clip(lower=p01, upper=p99, axis=1)
valores_recortados = int((X_recortado != X_df).sum().sum())
X_df = X_recortado
print("  Asimetría (skew) antes -> después del logaritmo:")
for v in ["gasto_anual", "compras_anuales"]:
    print(f"    {v}: {asimetria_antes[v]:.2f} -> {asimetria_despues[v]:.2f}")
print(f"  Valores recortados: {valores_recortados}")

fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df["gasto_anual"], bins=40, ax=ejes[0], color=ROSA)
ejes[0].set(title=f"Gasto anual original (asimetría = {asimetria_antes['gasto_anual']:.2f})",
            xlabel="Gasto anual (MXN)", ylabel="Clientes")
sns.histplot(X_df["gasto_anual"], bins=40, ax=ejes[1], color=MORADO)
ejes[1].set(title=f"log(1 + gasto) recortado P1-P99 (asimetría = "
                  f"{X_df['gasto_anual'].skew():.2f})",
            xlabel="log(1 + gasto anual)", ylabel="")
fig.suptitle("Tratamiento de atípicos y sesgo en el gasto anual", fontweight="bold")
guardar(fig, "05_transformacion_gasto.png")

# 3.8 Escalamiento
escalador = StandardScaler()
X = escalador.fit_transform(X_df)
print("\n3.8 Escalamiento con StandardScaler (media 0, desviación 1):")
print(pd.DataFrame(X, columns=VARS_MODELO).describe().loc[["mean", "std", "min", "max"]].round(3))
print("  Importancia: el gasto anual va de cientos a decenas de miles de pesos,")
print("  mientras que las compras van de 1 a ~20. Sin escalar, K-Means agruparía")
print("  casi solo por el gasto.")
print(f"\nRegistros finales: {len(df)} | Columnas: {df.shape[1]}")

# ==================================================================
# PARTE 4. IMPLEMENTACIÓN DEL ALGORITMO K-MEANS
# ==================================================================
titulo("PARTE 4. IMPLEMENTACIÓN DEL ALGORITMO K-MEANS")

filas = []
for k in range(2, 9):
    m = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(X)
    filas.append({
        "K": k,
        "inercia": round(m.inertia_, 2),
        "silueta": round(silhouette_score(X, m.labels_), 4),
        "davies_bouldin": round(davies_bouldin_score(X, m.labels_), 4),
    })
evaluacion = pd.DataFrame(filas)
evaluacion["reduccion_inercia_%"] = (-evaluacion["inercia"].pct_change() * 100).round(1)
evaluacion.to_csv(DIR_RESULTADOS / "evaluacion_k.csv", index=False)
print("\nEvaluación de K = 2 a 8:")
print(evaluacion.to_string(index=False))

K_FINAL = int(evaluacion.loc[evaluacion["silueta"].idxmax(), "K"])
print(f"\nK elegido: {K_FINAL}")
print("  Justificación: tiene el coeficiente de silueta más alto. En la gráfica")
print("  del codo la inercia baja 33% y 28% al pasar a K = 3 y K = 4; a partir de")
print("  K = 6 cada clúster extra reduce menos del 10%. K = 5 casi no mejora la")
print("  silueta. K = 3 tiene un Davies-Bouldin un poco menor, pero junta en")
print("  un solo grupo a clientes ocasionales e inactivos, que requieren")
print("  estrategias distintas (ver comparación de configuraciones).")

fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
for eje, col, nombre in [
        (ejes[0], "inercia", "Inercia (método del codo)"),
        (ejes[1], "silueta", "Coeficiente de silueta (mayor es mejor)"),
        (ejes[2], "davies_bouldin", "Índice Davies-Bouldin (menor es mejor)")]:
    eje.plot(evaluacion["K"], evaluacion[col], color=MORADO, lw=2, marker="o", ms=8)
    fila = evaluacion[evaluacion["K"] == K_FINAL].iloc[0]
    eje.scatter([K_FINAL], [fila[col]], s=170, facecolor="none",
                edgecolor=ROSA, lw=2.5, zorder=5, label=f"K = {K_FINAL}")
    eje.set_title(nombre)
    eje.set_xlabel("Número de clústeres (K)")
    eje.legend()
guardar(fig, "06_seleccion_k.png")

# Modelo final
modelo = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=10)
etiquetas = modelo.fit_predict(X)

# Centroides en unidades originales: se deshace el escalado y el logaritmo
cent_trans = pd.DataFrame(escalador.inverse_transform(modelo.cluster_centers_),
                          columns=VARS_MODELO)
centroides = cent_trans.copy()
centroides["gasto_anual"] = np.expm1(centroides["gasto_anual"])
centroides["compras_anuales"] = np.expm1(centroides["compras_anuales"])
centroides_z = pd.DataFrame(modelo.cluster_centers_, columns=VARS_MODELO)
print("\nCentroides en unidades escaladas (z):")
print(centroides_z.round(3))


# Nombrar cada clúster según su centroide (reglas de negocio)
def nombrar_clusters(cz):
    restantes = list(cz.index)
    nombres = {}
    reglas = [("Clientes premium", "gasto_anual", True),
              ("Cazadores de ofertas", "pct_compras_descuento", True),
              ("Clientes inactivos", "dias_ultima_compra", True)]
    for nombre, var, mayor in reglas[:len(restantes) - 1]:
        c = cz.loc[restantes, var].idxmax() if mayor else cz.loc[restantes, var].idxmin()
        nombres[c] = nombre
        restantes.remove(c)
    nombres[restantes[0]] = "Compradores ocasionales"
    return nombres


if K_FINAL > 4:
    raise SystemExit("Las reglas de nombres están pensadas para K <= 4")
mapa_nombres = nombrar_clusters(centroides_z)
ORDEN = [n for n in ["Clientes premium", "Cazadores de ofertas",
                     "Compradores ocasionales", "Clientes inactivos"]
         if n in mapa_nombres.values()]
renumerar = {c: ORDEN.index(n) for c, n in mapa_nombres.items()}
DESCRIPCION = {
    "Clientes premium": "Compran seguido, gastan mucho y casi no usan descuentos",
    "Cazadores de ofertas": "Compran con frecuencia, ticket bajo y casi siempre con promoción",
    "Compradores ocasionales": "Compran pocas veces al año, en fechas especiales o para regalo",
    "Clientes inactivos": "Llevan meses sin comprar y compraron muy poco en el año",
}
df["cluster"] = pd.Series(etiquetas).map(renumerar).values
df["nombre_cluster"] = pd.Series(etiquetas).map(mapa_nombres).values
df["descripcion_cluster"] = df["nombre_cluster"].map(DESCRIPCION)
centroides.index = centroides.index.map(renumerar)
centroides = centroides.sort_index()
centroides.insert(0, "nombre_cluster", ORDEN)
centroides_z.index = centroides_z.index.map(renumerar)
centroides_z = centroides_z.sort_index()

print(f"\nIteraciones hasta converger: {modelo.n_iter_}")
print(f"Inercia final: {modelo.inertia_:.2f}")
print("\nCentroides (unidades originales):")
print(centroides.round(2).to_string())
centroides.round(3).to_csv(DIR_RESULTADOS / "centroides.csv", index_label="cluster")

# Distancia de cada cliente a cada centroide (modelo.transform)
distancias = modelo.transform(X)
dist_ord = np.sort(distancias, axis=1)
df["distancia_centroide"] = dist_ord[:, 0].round(4)
df["margen_frontera"] = (dist_ord[:, 1] - dist_ord[:, 0]).round(4)
UMBRAL_FRONTERA = 0.3
df["caso_frontera"] = df["margen_frontera"] < UMBRAL_FRONTERA
print(f"\nClientes en la frontera entre dos clústeres (margen < {UMBRAL_FRONTERA}): "
      f"{int(df['caso_frontera'].sum())}")

conteo = df["nombre_cluster"].value_counts().reindex(ORDEN)
print("\nRegistros por clúster:")
print(conteo.to_string())

COLORES = dict(zip(ORDEN, [MORADO, ROSA, DORADO, VERDE]))
fig, ejes = plt.subplots(1, 2, figsize=(15, 5.8))
for nombre in ORDEN:
    g = df[df["nombre_cluster"] == nombre]
    ejes[0].scatter(g["dias_ultima_compra"], g["gasto_anual"], s=22, alpha=0.65,
                    color=COLORES[nombre], edgecolor="white", lw=0.4, label=nombre)
    ejes[1].scatter(g["compras_anuales"] + np.random.default_rng(1).uniform(-0.25, 0.25, len(g)),
                    g["pct_compras_descuento"], s=22, alpha=0.65,
                    color=COLORES[nombre], edgecolor="white", lw=0.4, label=nombre)
for eje, x, y in [(ejes[0], "dias_ultima_compra", "gasto_anual"),
                  (ejes[1], "compras_anuales", "pct_compras_descuento")]:
    eje.scatter(centroides[x], centroides[y], marker="X", s=330, color="black",
                edgecolor="white", lw=1.5, label="Centroides", zorder=5)
    eje.set(xlabel=ETIQUETA[x], ylabel=ETIQUETA[y])
ejes[0].set_yscale("log")
ejes[0].set_title("Recencia vs gasto anual (escala logarítmica)")
ejes[1].set_title("Frecuencia vs % de compras con descuento")
ejes[0].legend(fontsize=9)
fig.suptitle(f"Segmentos de clientes obtenidos con K-Means (K = {K_FINAL})",
             fontweight="bold")
guardar(fig, "07_clusters_dispersion.png")

# Comparación de configuraciones
modelo_k3 = KMeans(n_clusters=3, random_state=SEMILLA, n_init=10).fit(X)
X_sin = df[VARS_MODELO].values  # sin logaritmo ni escalado
modelo_sin = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=10).fit(X_sin)


def fila_config(nombre, k, lab):
    return {"configuracion": nombre, "clusteres": k,
            "registros_por_cluster": " / ".join(map(str, sorted(np.bincount(lab), reverse=True))),
            "silueta": round(silhouette_score(X, lab), 4),
            "davies_bouldin": round(davies_bouldin_score(X, lab), 4),
            "ARI_vs_modelo_final": round(adjusted_rand_score(modelo.labels_, lab), 4)}


comparacion = pd.DataFrame([
    fila_config(f"A: K={K_FINAL}, log + escalado (modelo final)", K_FINAL, modelo.labels_),
    fila_config("B: K=3, log + escalado", 3, modelo_k3.labels_),
    fila_config(f"C: K={K_FINAL}, datos SIN transformar ni escalar", K_FINAL, modelo_sin.labels_),
])
comparacion.to_csv(DIR_RESULTADOS / "comparacion_configuraciones.csv", index=False)
print("\nComparación de configuraciones:")
print(comparacion.to_string(index=False))
print("  (silueta y Davies-Bouldin se miden siempre sobre los datos transformados")
print("   y escalados para que las tres configuraciones sean comparables)")

cruce_k3 = pd.crosstab(df["nombre_cluster"], modelo_k3.labels_).reindex(ORDEN)
print("\nCómo se reparten los segmentos de K = 4 en los 3 grupos de K = 3:")
print(cruce_k3)
cruce_sin = pd.crosstab(df["nombre_cluster"], modelo_sin.labels_).reindex(ORDEN)
print("\nCómo se reparten los segmentos de K = 4 en el modelo sin escalar:")
print(cruce_sin)
centros_sin = pd.DataFrame(modelo_sin.cluster_centers_, columns=VARS_MODELO).round(1)
print("Centroides del modelo sin escalar:")
print(centros_sin)

# Estabilidad del modelo: se repite con 10 semillas distintas
ari_semillas = [adjusted_rand_score(
    modelo.labels_, KMeans(n_clusters=K_FINAL, random_state=s, n_init=10).fit_predict(X))
    for s in range(10)]
print(f"\nEstabilidad (ARI vs modelo final con 10 semillas): "
      f"mínimo {min(ari_semillas):.3f}, promedio {np.mean(ari_semillas):.3f}")

colores_k3 = [MORADO, DORADO, VERDE]
fig, ejes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
for nombre in ORDEN:
    g = df[df["nombre_cluster"] == nombre]
    ejes[0].scatter(g["dias_ultima_compra"], g["gasto_anual"], s=18, alpha=0.65,
                    color=COLORES[nombre], label=nombre)
gasto_k3 = df.groupby(modelo_k3.labels_)["gasto_anual"].mean().sort_values(ascending=False)
for i, (color, c) in enumerate(zip(colores_k3, gasto_k3.index), start=1):
    g = df[modelo_k3.labels_ == c]
    ejes[1].scatter(g["dias_ultima_compra"], g["gasto_anual"], s=18, alpha=0.65,
                    color=color, label=f"Grupo {i} ({len(g)} clientes)")
ejes[0].set(title=f"Configuración A: K = {K_FINAL}", xlabel=ETIQUETA["dias_ultima_compra"],
            ylabel=ETIQUETA["gasto_anual"], yscale="log")
ejes[1].set(title="Configuración B: K = 3", xlabel=ETIQUETA["dias_ultima_compra"])
ejes[0].legend(fontsize=9); ejes[1].legend(fontsize=9)
fig.suptitle("Comparación de configuraciones del modelo", fontweight="bold")
guardar(fig, "08_comparacion_k4_k3.png")

# ==================================================================
# PARTE 5. ANÁLISIS E INTERPRETACIÓN DE RESULTADOS
# ==================================================================
titulo("PARTE 5. ANÁLISIS E INTERPRETACIÓN DE RESULTADOS")

estad = (df.groupby("nombre_cluster")[NUMERICAS]
         .agg(["mean", "std", "min", "median", "max"]).round(2).reindex(ORDEN))
estad.to_csv(DIR_RESULTADOS / "estadisticas_por_cluster.csv")
print("\nEstadísticas descriptivas por clúster (variables del modelo):")
print(df.groupby("nombre_cluster")[VARS_MODELO].describe().round(1).T[ORDEN])

# Tabla comparativa de los grupos
tabla = df.groupby("nombre_cluster")[NUMERICAS].mean().T[ORDEN]
tabla["promedio_general"] = df[NUMERICAS].mean()
tabla = tabla.round(2)
tabla.to_csv(DIR_RESULTADOS / "tabla_comparativa.csv", index_label="variable")
print("\nTabla comparativa de los grupos (promedios):")
print(tabla)

# Aporte de cada segmento a las ventas
ventas = df.groupby("nombre_cluster")["gasto_anual"].sum().reindex(ORDEN)
pct_clientes = (conteo / len(df) * 100).round(1)
pct_ventas = (ventas / ventas.sum() * 100).round(1)
aporte = pd.DataFrame({"clientes": conteo, "pct_clientes": pct_clientes,
                       "ventas_mxn": ventas.round(0), "pct_ventas": pct_ventas})
aporte.to_csv(DIR_RESULTADOS / "aporte_ventas.csv", index_label="nombre_cluster")
print("\nAporte de cada segmento a las ventas del departamento:")
print(aporte)

# Perfil de los segmentos en variables categóricas (% por fila)
perfiles_cat = {}
for col in ["genero", "rango_edad", "familia_olfativa", "concentracion_preferida",
            "linea_preferida", "canal_preferido", "tarjeta_departamental", "sucursal"]:
    t = (pd.crosstab(df["nombre_cluster"], df[col], normalize="index") * 100).round(1).reindex(ORDEN)
    perfiles_cat[col] = t
    print(f"\n% por segmento - {col}:")
    print(t)
with open(DIR_RESULTADOS / "perfiles_categoricos.json", "w", encoding="utf-8") as f:
    json.dump({k: {i: {str(c): float(v) for c, v in fila.items()} for i, fila in t.iterrows()}
               for k, t in perfiles_cat.items()}, f, ensure_ascii=False, indent=2)

frontera_por_cluster = df.groupby("nombre_cluster")["caso_frontera"].sum().reindex(ORDEN)
print("\nCasos de frontera por segmento:")
print(frontera_por_cluster.to_string())

# Gráfica: perfil de los segmentos (promedio estandarizado)
perfil = ((df.groupby("nombre_cluster")[NUMERICAS].mean() - df[NUMERICAS].mean())
          / df[NUMERICAS].std()).T[ORDEN]
fig, eje = plt.subplots(figsize=(13, 5))
pos = np.arange(len(NUMERICAS))
ancho = 0.8 / len(ORDEN)
for i, nombre in enumerate(ORDEN):
    eje.bar(pos + (i - (len(ORDEN) - 1) / 2) * ancho, perfil[nombre], width=ancho * 0.95,
            color=COLORES[nombre], label=nombre)
eje.axhline(0, color=GRIS, lw=1)
eje.set_xticks(pos, [ETIQUETA[v] for v in NUMERICAS], rotation=25, ha="right")
eje.set_ylabel("Desviaciones respecto al promedio general")
eje.set_title("Perfil de cada segmento en las variables numéricas")
eje.legend(fontsize=9, ncol=2)
for v in VARS_MODELO:
    eje.get_xticklabels()[NUMERICAS.index(v)].set_fontweight("bold")
guardar(fig, "09_perfil_clusters.png")

# Gráfica: diagramas de caja por clúster
fig, ejes = plt.subplots(1, 4, figsize=(17, 4.6))
for eje, var in zip(ejes, VARS_MODELO):
    sns.boxplot(data=df, x="nombre_cluster", y=var, hue="nombre_cluster",
                palette=COLORES, ax=eje, legend=False, order=ORDEN)
    eje.set_title(ETIQUETA[var])
    eje.set_xlabel(""); eje.set_ylabel("")
    eje.set_xticks(range(len(ORDEN)), [n.replace(" ", "\n", 1) for n in ORDEN], fontsize=8.5)
    if var == "gasto_anual":
        eje.set_yscale("log")
fig.suptitle("Comparación de las variables del modelo por segmento", fontweight="bold")
guardar(fig, "10_boxplots_por_cluster.png")

# Gráfica: clientes vs ventas por segmento
fig, eje = plt.subplots(figsize=(9, 4.5))
pos = np.arange(len(ORDEN))
eje.bar(pos - 0.2, pct_clientes, width=0.38, color="#E4DFD8", label="% de clientes")
eje.bar(pos + 0.2, pct_ventas, width=0.38, color=[COLORES[n] for n in ORDEN],
        label="% de las ventas")
for i in range(len(ORDEN)):
    eje.text(i - 0.2, pct_clientes.iloc[i] + 1, f"{pct_clientes.iloc[i]:.0f}%", ha="center")
    eje.text(i + 0.2, pct_ventas.iloc[i] + 1, f"{pct_ventas.iloc[i]:.0f}%", ha="center",
             fontweight="bold")
eje.set_xticks(pos, ORDEN)
eje.set_ylim(0, max(pct_ventas.max(), pct_clientes.max()) * 1.15)
eje.set_ylabel("Porcentaje")
eje.set_title("Porcentaje de clientes contra porcentaje de las ventas")
eje.legend()
guardar(fig, "11_clientes_vs_ventas.png")

# Gráfica: preferencias por segmento
ORDEN_CAT = {"concentracion_preferida": ["Eau de Cologne", "Eau de Toilette",
                                         "Eau de Parfum", "Parfum"],
             "linea_preferida": ["Celebridad / masivo", "Diseñador", "Nicho"]}
fig, ejes = plt.subplots(1, 2, figsize=(15, 5.2))
for eje, col, tit in [(ejes[0], "concentracion_preferida", "Concentración preferida"),
                      (ejes[1], "linea_preferida", "Línea preferida")]:
    datos_cat = perfiles_cat[col][ORDEN_CAT[col]]
    datos_cat.plot(kind="barh", stacked=True, ax=eje, width=0.65,
                   color=["#ECE8FA", "#C9C3EE", "#9D94E0", "#6E63C2"][:datos_cat.shape[1]],
                   edgecolor="white")
    for i, fila in enumerate(datos_cat.values):
        acumulado = 0
        for j, v in enumerate(fila):
            if v >= 8:
                eje.text(acumulado + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8.5, color="#2b2440" if j < 3 else "white")
            acumulado += v
    eje.invert_yaxis()
    eje.set(title=f"{tit} por segmento (%)", xlabel="% de clientes del segmento", ylabel="")
    eje.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=len(ORDEN_CAT[col]), frameon=False)
    eje.set_xlim(0, 100)
fig.suptitle("Preferencias de producto de cada segmento", fontweight="bold")
guardar(fig, "12_preferencias_por_cluster.png")

# Guardar dataset procesado
columnas_salida = (["id_cliente", "genero", "edad", "rango_edad", "sucursal",
                    "canal_preferido", "canal_cod", "tarjeta_departamental", "tarjeta_cod",
                    "antiguedad_meses", "compras_anuales", "gasto_anual", "ticket_promedio",
                    "dias_ultima_compra", "pct_compras_descuento", "familia_olfativa",
                    "concentracion_preferida", "linea_preferida", "satisfaccion",
                    "cluster", "nombre_cluster", "descripcion_cluster",
                    "distancia_centroide", "margen_frontera", "caso_frontera"])
df[columnas_salida].to_csv(RUTA_PROCESADO, index=False, encoding="utf-8")
print(f"\nDataset procesado guardado en: {RUTA_PROCESADO.relative_to(RAIZ)}")

pd.DataFrame(bitacora, columns=["revision", "encontrados", "accion"]).to_csv(
    DIR_RESULTADOS / "bitacora_limpieza.csv", index=False)

resumen = {
    "registros_originales": int(registros_inicio),
    "registros": int(len(df)),
    "columnas_originales": int(df_original.shape[1]),
    "nulos_originales": nulos,
    "duplicados_originales": duplicados,
    "variables_modelo": VARS_MODELO,
    "k_final": K_FINAL,
    "orden_clusters": ORDEN,
    "iteraciones": int(modelo.n_iter_),
    "inercia": round(float(modelo.inertia_), 2),
    "silueta": round(float(silhouette_score(X, modelo.labels_)), 4),
    "davies_bouldin": round(float(davies_bouldin_score(X, modelo.labels_)), 4),
    "ari_estabilidad_min": round(float(min(ari_semillas)), 4),
    "ari_estabilidad_prom": round(float(np.mean(ari_semillas)), 4),
    "registros_por_cluster": {k: int(v) for k, v in conteo.items()},
    "pct_clientes": {k: float(v) for k, v in pct_clientes.items()},
    "pct_ventas": {k: float(v) for k, v in pct_ventas.items()},
    "ventas_totales": round(float(df["gasto_anual"].sum()), 2),
    "valores_recortados": valores_recortados,
    "registros_con_atipico": registros_con_atipico,
    "casos_frontera": int(df["caso_frontera"].sum()),
    "umbral_frontera": UMBRAL_FRONTERA,
    "frontera_por_cluster": {k: int(v) for k, v in frontera_por_cluster.items()},
    "centroides": centroides.round(2).to_dict(orient="records"),
    "centroides_z": centroides_z.round(3).assign(nombre_cluster=ORDEN).to_dict(orient="records"),
    "promedios": {n: {v: round(float(tabla.loc[v, n]), 2) for v in NUMERICAS} for n in ORDEN},
    "promedio_general": {v: round(float(tabla.loc[v, "promedio_general"]), 2) for v in NUMERICAS},
}
with open(DIR_RESULTADOS / "resumen.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=2)

print("\nHallazgos principales:")
prem = "Clientes premium"
print(f"  1. Los {conteo[prem]} clientes premium son el {pct_clientes[prem]:.0f}% de la base,")
print(f"     pero generan el {pct_ventas[prem]:.0f}% de las ventas del departamento.")
print(f"  2. Los cazadores de ofertas compran {tabla.loc['compras_anuales', 'Cazadores de ofertas']:.1f} "
      f"veces al año, pero con un ticket de ${tabla.loc['ticket_promedio', 'Cazadores de ofertas']:,.0f};")
print(f"     el {tabla.loc['pct_compras_descuento', 'Cazadores de ofertas']:.0f}% de sus compras llevan descuento.")
print(f"  3. Los clientes inactivos llevan en promedio "
      f"{tabla.loc['dias_ultima_compra', 'Clientes inactivos']:.0f} días sin comprar.")
print("\nAnálisis terminado.")
