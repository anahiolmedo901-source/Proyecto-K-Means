"""
Dashboard interactivo: Segmentación de clientes de perfumería con K-Means
Autora: Viviana Anahí Olmedo García

Ejecutar desde la carpeta raíz del proyecto (después de correr el análisis):

    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "procesado" / "clientes_perfumeria_clusters.csv"
RESULTADOS = RAIZ / "resultados"

# ------------------------------------------------------------------
# Identidad visual: tonos pastel (validados para daltonismo) y tipografía
# ------------------------------------------------------------------
TINTA, TINTA_SUAVE, LINEA, FONDO = "#2B2733", "#7A7383", "#E9E3DC", "#FFFFFF"
ORDEN = ["Clientes premium", "Cazadores de ofertas", "Compradores ocasionales",
         "Clientes inactivos"]
COLOR = dict(zip(ORDEN, ["#8C84D8", "#E07FA6", "#D9A447", "#4DAE90"]))
TINTE = dict(zip(ORDEN, ["#EEEBFA", "#FBE8F0", "#FAF0DC", "#E2F3EC"]))
CORTO = dict(zip(ORDEN, ["Premium", "Ofertas", "Ocasionales", "Inactivos"]))
SECUENCIAL = LinearSegmentedColormap.from_list("lavanda", ["#F7F5FC", "#C9C3EE", "#6E63C2"])
DIVERGENTE = LinearSegmentedColormap.from_list("rosa_lavanda", ["#D4668F", "#F4F1EE", "#6E63C2"])

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
CATEGORICAS = {"familia_olfativa": "Familia olfativa",
               "concentracion_preferida": "Concentración preferida",
               "linea_preferida": "Línea preferida",
               "canal_preferido": "Canal preferido",
               "tarjeta_departamental": "Tarjeta departamental",
               "genero": "Género", "rango_edad": "Rango de edad", "sucursal": "Sucursal"}
ORDEN_CAT = {"concentracion_preferida": ["Eau de Cologne", "Eau de Toilette",
                                         "Eau de Parfum", "Parfum"],
             "linea_preferida": ["Celebridad / masivo", "Diseñador", "Nicho"],
             "rango_edad": ["18-25", "26-35", "36-50", "51 o más"]}

# Solo se usan las fuentes instaladas para evitar avisos "findfont" en la consola
from matplotlib import font_manager
_instaladas = {f.name for f in font_manager.fontManager.ttflist}
FUENTES = [f for f in ["Segoe UI", "DM Sans", "Helvetica Neue", "Arial", "DejaVu Sans"]
           if f in _instaladas] or ["DejaVu Sans"]
plt.rcParams.update({
    "font.family": FUENTES,
    "font.size": 9.5, "text.color": TINTA, "axes.labelcolor": TINTA_SUAVE,
    "axes.edgecolor": LINEA, "axes.linewidth": 0.8, "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": "#F0EBE5", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "legend.frameon": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.axisbelow": True,
})

st.set_page_config(page_title="Perfumería · Segmentación de clientes", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

html, body, .stMarkdown, .stText, p, li, label, input, textarea,
[data-baseweb="select"] div, [data-baseweb="tag"] span, [data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"], [data-testid="stSidebar"] {
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
}
.panel-nota, .lista, .lista *, .segmento p, .segmento .lema, .cifra span, .kpi .nombre,
.kpi .nota, .bajada, .seccion p, .nota-pie, .antetitulo, .seccion .num, .submarca,
.etiqueta-lateral { font-family: 'DM Sans', 'Segoe UI', sans-serif !important; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
h1, h2, h3, h4, .serif { font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 500 !important; letter-spacing: -0.01em; color: #2B2733; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

/* Barra lateral */
[data-testid="stSidebar"] { background: #F3EEE8; border-right: 1px solid #E9E3DC; }
[data-testid="stSidebar"] .marca { font-family: 'Fraunces', Georgia, serif; font-size: 1.45rem;
    line-height: 1.15; color: #2B2733; margin: 0 0 .2rem 0; }
[data-testid="stSidebar"] .submarca { color: #7A7383; font-size: .82rem; margin-bottom: 1.4rem; }
.etiqueta-lateral { text-transform: uppercase; letter-spacing: .12em; font-size: .7rem;
    color: #7A7383; font-weight: 600; margin: 1.2rem 0 .4rem 0; }
[data-baseweb="tag"] { background: #E6E1F7 !important; border-radius: 999px !important; }
[data-baseweb="tag"] span { color: #3F3780 !important; }
[data-baseweb="tag"] svg { fill: #3F3780 !important; }

/* Encabezado */
.antetitulo { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem;
    color: #8C84D8; font-weight: 600; margin-bottom: .35rem; }
.titulo { font-family: 'Fraunces', Georgia, serif; font-size: 2.55rem; line-height: 1.1;
    color: #2B2733; margin: 0 0 .6rem 0; font-weight: 500; }
.bajada { color: #5E5867; font-size: 1.02rem; max-width: 760px; line-height: 1.55; }

/* Tarjetas de indicadores */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.6rem 0 .6rem 0; }
.kpi { border-radius: 16px; padding: 18px 20px 16px 20px; border: 1px solid rgba(43,39,51,.05); }
.kpi .nombre { font-size: .78rem; color: #5E5867; font-weight: 500; }
.kpi .valor { font-family: 'Fraunces', Georgia, serif; font-size: 2.05rem; color: #2B2733;
    line-height: 1.15; margin-top: 6px; }
.kpi .nota { font-size: .76rem; color: #7A7383; margin-top: 4px; }

/* Secciones */
.seccion { margin: 2.4rem 0 .9rem 0; padding-top: 1.3rem; border-top: 1px solid #E9E3DC; }
.seccion .num { text-transform: uppercase; letter-spacing: .14em; font-size: .7rem;
    color: #7A7383; font-weight: 600; }
.seccion h2 { font-size: 1.65rem; margin: .25rem 0 .2rem 0; padding: 0; }
.seccion p { color: #5E5867; margin: 0; font-size: .93rem; }

/* Paneles de gráficas */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .panel) {
    background: #FFFFFF; border-radius: 16px !important; border: 1px solid #EDE7E0 !important;
    padding: 1.2rem 1.3rem 1rem 1.3rem; box-shadow: 0 1px 2px rgba(43,39,51,.03);
}
[data-testid="stExpander"] details { background: #FFFFFF; border-radius: 16px; border-color: #EDE7E0; }
.panel { font-family: 'Fraunces', Georgia, serif; font-size: 1.08rem; color: #2B2733; margin-bottom: .1rem; }
.panel-nota { color: #7A7383; font-size: .8rem; margin-bottom: .3rem; }

/* Tarjetas de segmentos */
.segmentos { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
.segmento { border-radius: 16px; padding: 20px 22px; border-top: 4px solid; }
.segmento .nombre { font-family: 'Fraunces', Georgia, serif; font-size: 1.3rem; color: #2B2733; }
.segmento .lema { color: #5E5867; font-size: .86rem; margin: 2px 0 12px 0; }
.segmento .cifras { display: flex; gap: 26px; margin-bottom: 12px; }
.segmento .cifra b { display: block; font-family: 'Fraunces', Georgia, serif; font-weight: 500;
    font-size: 1.35rem; color: #2B2733; }
.segmento .cifra span { font-size: .74rem; color: #7A7383; }
.segmento p { font-size: .88rem; color: #3F3A47; line-height: 1.55; margin: 0; }

/* Hallazgos */
.lista { counter-reset: n; }
.lista .item { position: relative; padding: 0 0 14px 40px; color: #3F3A47; font-size: .92rem; line-height: 1.55; }
.lista .item::before { counter-increment: n; content: counter(n, decimal-leading-zero);
    position: absolute; left: 0; top: 0; font-family: 'Fraunces', Georgia, serif; color: #8C84D8; font-size: 1.05rem; }
.nota-pie { color: #7A7383; font-size: .8rem; margin-top: 1.6rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATOS)
    with open(RESULTADOS / "resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)
    evaluacion = pd.read_csv(RESULTADOS / "evaluacion_k.csv")
    comparacion = pd.read_csv(RESULTADOS / "comparacion_configuraciones.csv")
    bitacora = pd.read_csv(RESULTADOS / "bitacora_limpieza.csv")
    return df, resumen, evaluacion, comparacion, bitacora


df, resumen, evaluacion, comparacion, bitacora = cargar_datos()


def pesos(valor):
    return f"${valor:,.0f}"


def figura(ancho=6, alto=3.8):
    return plt.subplots(figsize=(ancho, alto))


def seccion(numero, titulo, texto=""):
    st.markdown(f"<div class='seccion'><div class='num'>{numero}</div><h2>{titulo}</h2>"
                f"<p>{texto}</p></div>", unsafe_allow_html=True)


def panel(titulo, nota=""):
    st.markdown(f"<div class='panel'>{titulo}</div>"
                + (f"<div class='panel-nota'>{nota}</div>" if nota else ""),
                unsafe_allow_html=True)


def mostrar(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def kpis(items):
    html = "".join(f"<div class='kpi' style='background:{tinte}'><div class='nombre'>{nombre}</div>"
                   f"<div class='valor'>{valor}</div><div class='nota'>{nota}</div></div>"
                   for nombre, valor, nota, tinte in items)
    st.markdown(f"<div class='kpis'>{html}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Barra lateral: navegación y filtros interactivos
# ------------------------------------------------------------------
st.sidebar.markdown("<div class='marca'>Perfumería</div>"
                    "<div class='submarca'>Segmentación de clientes con K&#8209;Means</div>",
                    unsafe_allow_html=True)
pagina = st.sidebar.radio("Página", ["Análisis general", "Segmentación K-Means"],
                          label_visibility="collapsed")

st.sidebar.markdown("<div class='etiqueta-lateral'>Filtros</div>", unsafe_allow_html=True)
filtro_cluster = st.sidebar.multiselect("Segmento", ORDEN, default=ORDEN)
sucursales = sorted(df["sucursal"].unique())
filtro_sucursal = st.sidebar.multiselect("Sucursal", sucursales, default=sucursales)
generos = sorted(df["genero"].unique())
filtro_genero = st.sidebar.multiselect("Género", generos, default=generos)
emin, emax = int(df["edad"].min()), int(df["edad"].max())
filtro_edad = st.sidebar.slider("Rango de edad", emin, emax, (emin, emax))

datos = df[df["nombre_cluster"].isin(filtro_cluster)
           & df["sucursal"].isin(filtro_sucursal)
           & df["genero"].isin(filtro_genero)
           & df["edad"].between(*filtro_edad)]

st.sidebar.markdown("<div class='etiqueta-lateral'>Datos</div>", unsafe_allow_html=True)
st.sidebar.caption(f"Clientes mostrados: **{len(datos):,}** de {len(df):,}")
st.sidebar.caption("Datos simulados del departamento de perfumería de una tienda "
                   "departamental ficticia (`src/generar_dataset.py`).")
st.sidebar.caption("Viviana Anahí Olmedo García · IDGS 9-2")

# ------------------------------------------------------------------
# Encabezado y KPI (comunes a las dos páginas)
# ------------------------------------------------------------------
numero_pagina = "Página 1" if pagina == "Análisis general" else "Página 2"
st.markdown(
    f"<div class='antetitulo'>Departamento de perfumería · {numero_pagina}</div>"
    "<div class='titulo'>Segmentación de clientes del departamento de perfumería</div>"
    "<div class='bajada'>Agrupamiento con K-Means de 1,500 clientes de una tienda departamental "
    "según qué tan reciente, frecuente y costosa es su compra y qué tanto usan descuentos, "
    "para diseñar una estrategia de venta por segmento.</div>", unsafe_allow_html=True)

if datos.empty:
    st.warning("Ningún cliente cumple con los filtros seleccionados. "
               "Ajusta los filtros de la barra lateral.")
    st.stop()

ventas = datos["gasto_anual"].sum()
kpis([
    ("Clientes", f"{len(datos):,}", f"{len(datos) / len(df):.0%} de la base", TINTE[ORDEN[0]]),
    ("Ventas anuales", f"${ventas / 1e6:,.2f} M", "pesos mexicanos", TINTE[ORDEN[1]]),
    ("Ticket promedio", pesos(ventas / datos["compras_anuales"].sum()), "por compra",
     TINTE[ORDEN[2]]),
    ("Compras con descuento", f"{datos['pct_compras_descuento'].mean():.1f}%",
     "promedio por cliente", TINTE[ORDEN[3]]),
])

# ==================================================================
# PÁGINA 1. ANÁLISIS GENERAL
# ==================================================================
if pagina == "Análisis general":
    seccion("Página 1", "Análisis general del dataset",
            "Cómo son los clientes y cómo compran, antes de aplicar el modelo.")

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        panel("Distribución de una variable", "La línea punteada marca el promedio.")
        var = st.selectbox("Variable", NUMERICAS, index=3, format_func=ETIQUETA.get)
        fig, eje = figura()
        sns.histplot(datos[var], bins=30, color="#B7B0E6", edgecolor=FONDO, linewidth=1.2,
                     alpha=1, ax=eje, discrete=(var == "satisfaccion"))
        media = datos[var].mean()
        eje.axvline(media, color="#6E63C2", lw=1.6, ls=(0, (4, 3)))
        eje.text(media, eje.get_ylim()[1] * 0.96, f"  promedio {media:,.1f}",
                 color="#4A4190", fontsize=8.5, va="top")
        eje.set_xlabel(ETIQUETA[var]); eje.set_ylabel("Clientes")
        eje.grid(axis="x", visible=False)
        mostrar(fig)
    with c2, st.container(border=True):
        panel("Clientes por categoría", "Número y porcentaje de clientes.")
        cat = st.selectbox("Variable categórica", list(CATEGORICAS), format_func=CATEGORICAS.get)
        conteo = datos[cat].value_counts().sort_values()
        fig, eje = figura()
        eje.barh(conteo.index.astype(str), conteo.values, color="#B7B0E6", height=0.62,
                 edgecolor=FONDO, linewidth=2)
        for i, v in enumerate(conteo.values):
            eje.text(v, i, f"  {v}  ({v / len(datos):.0%})", va="center", fontsize=8.5,
                     color=TINTA)
        eje.set_xlim(0, conteo.max() * 1.25)
        eje.set_xlabel("Clientes")
        eje.grid(axis="y", visible=False)
        mostrar(fig)

    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True):
        panel("Matriz de correlación", "Rosa: relación negativa · lavanda: relación positiva.")
        fig, eje = figura(6.2, 4.9)
        etiquetas = [ETIQUETA[v].split(" (")[0] for v in NUMERICAS]
        sns.heatmap(datos[NUMERICAS].corr(), annot=True, fmt=".2f", cmap=DIVERGENTE,
                    vmin=-1, vmax=1, center=0, ax=eje, annot_kws={"size": 7.5},
                    linewidths=1.5, linecolor=FONDO, cbar=False,
                    xticklabels=etiquetas, yticklabels=etiquetas)
        eje.grid(False)
        eje.tick_params(axis="x", rotation=40, labelsize=8)
        eje.tick_params(axis="y", labelsize=8)
        plt.setp(eje.get_xticklabels(), ha="right")
        mostrar(fig)
    with c4, st.container(border=True):
        panel("Ventas por sucursal y canal", "Millones de pesos al año.")
        tabla_v = (datos.groupby(["sucursal", "canal_preferido"])["gasto_anual"].sum()
                   .unstack(fill_value=0) / 1e6)
        fig, eje = figura(6, 4.55)
        colores_canal = {"Tienda física": "#B7B0E6", "En línea": "#F0B3C8"}
        base = np.zeros(len(tabla_v))
        for canal in [c for c in ["Tienda física", "En línea"] if c in tabla_v.columns]:
            eje.bar(tabla_v.index, tabla_v[canal], bottom=base, width=0.58,
                    color=colores_canal[canal], edgecolor=FONDO, linewidth=2, label=canal)
            base += tabla_v[canal].values
        for i, total in enumerate(base):
            eje.text(i, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8.5)
        eje.set_ylabel("Ventas (millones de MXN)")
        eje.set_ylim(0, base.max() * 1.15)
        eje.grid(axis="x", visible=False)
        eje.legend(loc="upper right", fontsize=8.5)
        mostrar(fig)

    with st.container(border=True):
        panel("Estadísticas descriptivas", "Variables numéricas de los clientes filtrados.")
        estad = datos[NUMERICAS].describe().T.round(2).rename(index=ETIQUETA)
        estad.columns = ["Clientes", "Media", "Desv. est.", "Mínimo", "25%", "Mediana",
                         "75%", "Máximo"]
        st.dataframe(estad, width="stretch")

    with st.expander("Limpieza aplicada a los datos originales"):
        st.markdown(
            f"De **{resumen['registros_originales']}** registros originales quedaron "
            f"**{resumen['registros']}** clientes: se eliminaron "
            f"{resumen['duplicados_originales']} duplicados y se trataron "
            f"{resumen['nulos_originales']} valores nulos.")
        st.dataframe(bitacora.rename(columns={"revision": "Revisión",
                                              "encontrados": "Encontrados",
                                              "accion": "Acción"}),
                     width="stretch", hide_index=True)

# ==================================================================
# PÁGINA 2. SEGMENTACIÓN MEDIANTE K-MEANS
# ==================================================================
else:
    seccion("Página 2", "Segmentación mediante K-Means",
            "Variables del modelo: días desde la última compra, compras al año, gasto anual y "
            "% de compras con descuento (gasto y compras con log(1 + x), recorte P1-P99 y "
            "StandardScaler).")

    kpis([
        ("Número de clústeres", str(resumen["k_final"]), "elegido entre K = 2 y 8", "#F3EEE8"),
        ("Coeficiente de silueta", f"{resumen['silueta']:.3f}", "mayor es mejor", "#F3EEE8"),
        ("Índice Davies-Bouldin", f"{resumen['davies_bouldin']:.3f}", "menor es mejor", "#F3EEE8"),
        ("Estabilidad (ARI)", f"{resumen['ari_estabilidad_prom']:.3f}",
         "promedio con 10 semillas", "#F3EEE8"),
    ])

    presentes = [c for c in ORDEN if c in datos["nombre_cluster"].unique()]
    etiquetas_x = [c.replace(" ", "\n", 1) for c in presentes]

    c1, c2 = st.columns(2, gap="medium")
    with c1, st.container(border=True):
        panel("Clientes por segmento", "Número de clientes y porcentaje del total filtrado.")
        conteo = datos["nombre_cluster"].value_counts().reindex(presentes)
        fig, eje = figura(6, 4.25)
        eje.bar(range(len(conteo)), conteo.values, color=[COLOR[c] for c in presentes],
                width=0.6, edgecolor=FONDO, linewidth=2)
        for i, v in enumerate(conteo.values):
            eje.text(i, v, f"{v}\n{v / conteo.sum():.0%}", ha="center", va="bottom",
                     fontsize=8.5, linespacing=1.3)
        eje.set_xticks(range(len(conteo)), etiquetas_x)
        eje.set_ylabel("Clientes")
        eje.set_ylim(0, conteo.max() * 1.22)
        eje.grid(axis="x", visible=False)
        mostrar(fig)
    with c2, st.container(border=True):
        panel("Dispersión de los agrupamientos", "La cruz marca el centro de cada segmento.")
        a, b = st.columns(2)
        ejex = a.selectbox("Eje X", NUMERICAS, index=5, format_func=ETIQUETA.get, key="x")
        ejey = b.selectbox("Eje Y", NUMERICAS, index=3, format_func=ETIQUETA.get, key="y")
        fig, eje = figura(6, 3.75)
        for c in presentes:
            g = datos[datos["nombre_cluster"] == c]
            eje.scatter(g[ejex], g[ejey], s=16, alpha=0.7, color=COLOR[c], label=CORTO[c],
                        edgecolor=FONDO, linewidth=0.4)
        centros = df.groupby("nombre_cluster")[[ejex, ejey]].mean().reindex(presentes)
        eje.scatter(centros[ejex], centros[ejey], marker="X", s=150, color=TINTA,
                    edgecolor=FONDO, lw=1.6, zorder=5, label="Centro")
        if ejey == "gasto_anual":
            eje.set_yscale("log")
        eje.set_xlabel(ETIQUETA[ejex]); eje.set_ylabel(ETIQUETA[ejey])
        eje.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                   handletextpad=0.2, columnspacing=0.9)
        mostrar(fig)

    c3, c4 = st.columns(2, gap="medium")
    with c3, st.container(border=True):
        panel("Clientes contra ventas", "Qué parte de la base es cada segmento y cuánto vende.")
        pc = datos["nombre_cluster"].value_counts(normalize=True).reindex(presentes) * 100
        pv = (datos.groupby("nombre_cluster")["gasto_anual"].sum().reindex(presentes)
              / datos["gasto_anual"].sum() * 100)
        fig, eje = figura(6, 4.25)
        pos = np.arange(len(presentes))
        eje.bar(pos - 0.2, pc, width=0.38, color="#E4DFD8", edgecolor=FONDO, linewidth=2,
                label="% de clientes")
        eje.bar(pos + 0.2, pv, width=0.38, color=[COLOR[c] for c in presentes],
                edgecolor=FONDO, linewidth=2, label="% de las ventas")
        for i in range(len(presentes)):
            eje.text(i - 0.2, pc.iloc[i] + 1, f"{pc.iloc[i]:.0f}%", ha="center", fontsize=8.5,
                     color=TINTA_SUAVE)
            eje.text(i + 0.2, pv.iloc[i] + 1, f"{pv.iloc[i]:.0f}%", ha="center", fontsize=8.5,
                     fontweight="bold")
        eje.set_xticks(pos, etiquetas_x)
        eje.set_ylim(0, max(pc.max(), pv.max()) * 1.18)
        eje.set_ylabel("Porcentaje")
        eje.grid(axis="x", visible=False)
        eje.legend(fontsize=8.5, loc="upper right")
        mostrar(fig)
    with c4, st.container(border=True):
        panel("Comparación de una variable por segmento", "Mediana, rango intercuartílico y extremos.")
        var = st.selectbox("Variable a comparar", NUMERICAS, index=6, format_func=ETIQUETA.get)
        fig, eje = figura(6, 3.75)
        sns.boxplot(data=datos, x="nombre_cluster", y=var, hue="nombre_cluster",
                    order=presentes, palette=COLOR, legend=False, ax=eje, width=0.55,
                    linewidth=1, linecolor=TINTA_SUAVE,
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": TINTA_SUAVE, "markeredgecolor": "none"})
        eje.set_xticks(range(len(presentes)), etiquetas_x)
        eje.set_xlabel(""); eje.set_ylabel(ETIQUETA[var])
        eje.grid(axis="x", visible=False)
        if var == "gasto_anual":
            eje.set_yscale("log")
        mostrar(fig)

    c5, c6 = st.columns(2, gap="medium")
    with c5, st.container(border=True):
        panel("Perfil de los segmentos",
              "Distancia de cada segmento al promedio general, en desviaciones estándar.")
        perfil = ((datos.groupby("nombre_cluster")[NUMERICAS].mean() - df[NUMERICAS].mean())
                  / df[NUMERICAS].std()).T[presentes]
        fig, eje = figura(6.2, 4.6)
        pos = np.arange(len(NUMERICAS))
        alto = 0.8 / len(presentes)
        for i, c in enumerate(presentes):
            eje.barh(pos + (i - (len(presentes) - 1) / 2) * alto, perfil[c], height=alto,
                     color=COLOR[c], edgecolor=FONDO, linewidth=0.8, label=CORTO[c])
        eje.axvline(0, color=TINTA_SUAVE, lw=0.9)
        eje.set_yticks(pos, [ETIQUETA[v] for v in NUMERICAS], fontsize=8.5)
        eje.invert_yaxis()
        eje.set_xlabel("Desviaciones respecto al promedio")
        eje.grid(axis="y", visible=False)
        eje.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0),
                   handletextpad=0.3, columnspacing=0.9)
        mostrar(fig)
    with c6, st.container(border=True):
        panel("Preferencias por segmento", "Porcentaje de los clientes de cada segmento.")
        cat = st.selectbox("Preferencia", ["concentracion_preferida", "linea_preferida",
                                           "familia_olfativa", "canal_preferido",
                                           "tarjeta_departamental", "rango_edad"],
                           format_func=CATEGORICAS.get)
        tabla = (pd.crosstab(datos["nombre_cluster"], datos[cat], normalize="index") * 100
                 ).reindex(presentes)
        if cat in ORDEN_CAT:
            tabla = tabla[[c for c in ORDEN_CAT[cat] if c in tabla.columns]]
        tabla.index = [CORTO[c] for c in tabla.index]
        fig, eje = figura(6.2, 3.95)
        sns.heatmap(tabla, annot=True, fmt=".0f", cmap=SECUENCIAL, cbar=False, ax=eje,
                    linewidths=2, linecolor=FONDO, vmin=0, vmax=max(60, tabla.values.max()),
                    annot_kws={"size": 9})
        eje.set_xlabel(""); eje.set_ylabel("")
        eje.tick_params(axis="x", rotation=0, labelsize=8.5)
        eje.tick_params(axis="y", rotation=0, labelsize=8.5)
        eje.grid(False)
        mostrar(fig)

    with st.container(border=True):
        panel("Estadísticas descriptivas por segmento",
              "Elige las variables; cada columna es un segmento.")
        variables_tabla = st.multiselect("Variables", NUMERICAS,
                                         default=resumen["variables_modelo"],
                                         format_func=ETIQUETA.get, label_visibility="collapsed")
        if variables_tabla:
            medidas = {"mean": "Media", "std": "Desv. est.", "min": "Mínimo",
                       "median": "Mediana", "max": "Máximo"}
            estad = (datos.groupby("nombre_cluster")[variables_tabla]
                     .agg(list(medidas)).round(2).reindex(presentes).T)
            estad.index = [f"{ETIQUETA[v]} · {medidas[m]}" for v, m in estad.index]
            estad.columns.name = None
            st.dataframe(estad, width="stretch")

    c7, c8 = st.columns(2, gap="medium")
    with c7, st.container(border=True):
        panel("Selección de K", "Inercia, silueta y Davies-Bouldin para K = 2 a 8.")
        tabla_k = evaluacion.rename(columns={
            "inercia": "Inercia", "silueta": "Silueta", "davies_bouldin": "Davies-Bouldin",
            "reduccion_inercia_%": "Reducción %"})
        # Columna de texto: K = 2 no tiene reducción porque no hay K anterior
        tabla_k["Reducción %"] = tabla_k["Reducción %"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.1f}%")
        st.dataframe(tabla_k, width="stretch", hide_index=True)
    with c8, st.container(border=True):
        panel("Comparación de configuraciones",
              f"{resumen['casos_frontera']} clientes quedaron casi a la misma distancia de dos "
              "centroides (casos de frontera, calculados con modelo.transform()).")
        st.dataframe(comparacion[["configuracion", "registros_por_cluster", "silueta",
                                  "davies_bouldin"]].rename(columns={
                         "configuracion": "Configuración", "registros_por_cluster": "Clientes por clúster",
                         "silueta": "Silueta", "davies_bouldin": "Davies-Bouldin"}),
                     width="stretch", hide_index=True)

    seccion("Resultados", "Descripción de los grupos identificados",
            "Cifras calculadas sobre los 1,500 clientes; no cambian con los filtros.")
    prom = resumen["promedios"]
    n = resumen["registros_por_cluster"]
    lemas = {
        "Clientes premium": "Leales, de alto gasto y sin descuentos",
        "Cazadores de ofertas": "Frecuentes, de ticket bajo y sensibles al precio",
        "Compradores ocasionales": "Pocas compras al año, probablemente para regalo",
        "Clientes inactivos": "Llevan meses sin regresar",
    }
    textos = {
        "Clientes premium": "Prefieren Eau de Parfum, Parfum y marcas de nicho; casi no esperan "
                            "promociones y la mayoría tiene tarjeta departamental.",
        "Cazadores de ofertas": "El 69% tiene entre 18 y 35 años. Prefieren Eau de Toilette y "
                                "líneas masivas o de celebridad, y son quienes más compran en línea.",
        "Compradores ocasionales": "Compran de 2 a 4 veces al año con ticket medio y prefieren "
                                   "perfumes de diseñador. Es el único segmento con más hombres.",
        "Clientes inactivos": "Compraron una o dos veces en el año, son los de mayor edad y "
                              "tienen la satisfacción más baja.",
    }
    tarjetas = ""
    for c in ORDEN:
        p = prom[c]
        tarjetas += (
            f"<div class='segmento' style='background:{TINTE[c]}; border-color:{COLOR[c]}'>"
            f"<div class='nombre'>{c}</div><div class='lema'>{lemas[c]}</div>"
            "<div class='cifras'>"
            f"<div class='cifra'><b>{n[c]}</b><span>clientes</span></div>"
            f"<div class='cifra'><b>{resumen['pct_ventas'][c]:.0f}%</b><span>de las ventas</span></div>"
            f"<div class='cifra'><b>{pesos(p['gasto_anual'])}</b><span>gasto anual</span></div>"
            f"<div class='cifra'><b>{p['dias_ultima_compra']:.0f}</b><span>días sin comprar</span></div>"
            "</div>"
            f"<p>{p['compras_anuales']:.1f} compras al año, ticket de {pesos(p['ticket_promedio'])} y "
            f"{p['pct_compras_descuento']:.0f}% de compras con descuento. {textos[c]}</p></div>")
    st.markdown(f"<div class='segmentos'>{tarjetas}</div>", unsafe_allow_html=True)

    seccion("Conclusiones", "Principales hallazgos y recomendaciones")
    pv, pc = resumen["pct_ventas"], resumen["pct_clientes"]
    hallazgos = [
        f"Los clientes premium son solo el {pc['Clientes premium']:.0f}% de la base, pero generan "
        f"el <b>{pv['Clientes premium']:.0f}% de las ventas</b> del departamento.",
        f"Los cazadores de ofertas compran {prom['Cazadores de ofertas']['compras_anuales']:.1f} "
        f"veces al año, pero el {prom['Cazadores de ofertas']['pct_compras_descuento']:.0f}% de "
        "sus compras lleva descuento y su ticket es el más bajo.",
        f"Uno de cada cinco clientes está inactivo: {n['Clientes inactivos']} clientes con "
        f"{prom['Clientes inactivos']['dias_ultima_compra']:.0f} días promedio sin comprar, que "
        f"aportan solo el {pv['Clientes inactivos']:.0f}% de las ventas.",
        "La sucursal casi no cambia entre segmentos: el comportamiento de compra depende del "
        "cliente, no de la tienda.",
    ]
    recomendaciones = [
        "<b>Premium:</b> programa de lealtad con preventa de lanzamientos, muestras de nicho y "
        "atención personalizada, en lugar de descuentos.",
        "<b>Cazadores de ofertas:</b> concentrar las promociones en línea y en temporadas "
        "(Buen Fin, Día de las Madres) y ofrecer sets para subir el ticket.",
        "<b>Ocasionales:</b> campañas de regalo antes de San Valentín, Día del Padre y Navidad.",
        "<b>Inactivos:</b> campaña de reactivación con un cupón según su familia olfativa "
        "preferida y medir quién regresa.",
    ]
    h, r = st.columns(2, gap="medium")
    for col, titulo, items in [(h, "Hallazgos", hallazgos), (r, "Recomendaciones", recomendaciones)]:
        with col, st.container(border=True):
            panel(titulo)
            st.markdown("<div class='lista'>" + "".join(f"<div class='item'>{t}</div>" for t in items)
                        + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='nota-pie'>Los datos son simulados: los resultados ilustran la "
                "metodología y no describen a una tienda real.</div>", unsafe_allow_html=True)
