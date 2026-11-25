"""
Generación del dataset propio (simulado) del proyecto.

Simula la base de clientes del departamento de perfumería de una tienda
departamental ficticia durante un año de ventas. Los datos NO pertenecen a
ninguna empresa real: se generan con reglas de negocio razonables y una
semilla fija para que el resultado sea siempre el mismo.

Ejecutar desde la carpeta raíz del proyecto:

    python src/generar_dataset.py

Genera:
    data/original/clientes_perfumeria.csv

Para que la etapa de limpieza tenga trabajo real, el archivo incluye a
propósito problemas típicos de una base capturada en tienda: valores nulos,
registros duplicados, texto escrito de distintas formas, edades imposibles,
recencias negativas y montos guardados como texto con "$" y comas.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "original" / "clientes_perfumeria.csv"
SEMILLA = 2026
N_CLIENTES = 1500

rng = np.random.default_rng(SEMILLA)

# Perfiles de compra que existen en la tienda (el algoritmo NO los conoce;
# no se guardan en el archivo).
PERFILES = {
    #               prop  compras ticket  ticket_sd recencia desc  edad edad_sd  pF   tarjeta satisf antig
    "premium":     (0.17, 10.0,  2600,   650,      20,      0.14, 42,  9,       0.62, 0.80,   4.5,   62),
    "ofertas":     (0.30, 7.0,   900,    260,      38,      0.70, 31,  8,       0.68, 0.55,   4.0,   30),
    "ocasional":   (0.33, 3.0,   1750,   480,      100,     0.28, 38,  12,      0.45, 0.25,   3.8,   24),
    "inactivo":    (0.20, 1.3,   1250,   420,      270,     0.38, 46,  13,      0.55, 0.30,   3.1,   44),
}
FAMILIAS = ["Floral", "Amaderada", "Oriental", "Cítrica", "Fresca"]
PROB_FAMILIA = {
    "premium":   [0.22, 0.30, 0.33, 0.07, 0.08],
    "ofertas":   [0.38, 0.10, 0.10, 0.17, 0.25],
    "ocasional": [0.30, 0.20, 0.18, 0.14, 0.18],
    "inactivo":  [0.32, 0.20, 0.15, 0.15, 0.18],
}
CONCENTRACIONES = ["Eau de Cologne", "Eau de Toilette", "Eau de Parfum", "Parfum"]
PROB_CONC = {
    "premium":   [0.02, 0.15, 0.53, 0.30],
    "ofertas":   [0.20, 0.55, 0.22, 0.03],
    "ocasional": [0.08, 0.40, 0.42, 0.10],
    "inactivo":  [0.12, 0.45, 0.36, 0.07],
}
PROB_LINEA = {  # Diseñador, Nicho, Celebridad / masivo
    "premium":   [0.55, 0.40, 0.05],
    "ofertas":   [0.45, 0.03, 0.52],
    "ocasional": [0.70, 0.08, 0.22],
    "inactivo":  [0.62, 0.06, 0.32],
}
PROB_ONLINE = {"premium": 0.25, "ofertas": 0.55, "ocasional": 0.35, "inactivo": 0.30}
SUCURSALES = ["Centro", "Norte", "Sur", "Poniente"]

filas = []
nombres = list(PERFILES)
props = np.array([PERFILES[p][0] for p in nombres])
asignados = rng.choice(nombres, size=N_CLIENTES, p=props / props.sum())

for i, perfil in enumerate(asignados, start=1):
    (_, compras_m, ticket_m, ticket_sd, rec_m, desc_m, edad_m, edad_sd,
     p_fem, p_tarjeta, satisf_m, antig_m) = PERFILES[perfil]

    compras = max(1, int(rng.poisson(compras_m)))
    ticket = max(250.0, rng.normal(ticket_m, ticket_sd))
    gasto = compras * ticket * rng.lognormal(0, 0.10)
    recencia = int(np.clip(rng.gamma(4.0, rec_m / 4.0), 1, 365))
    descuento = np.clip(rng.beta(desc_m * 9, (1 - desc_m) * 9), 0, 1)
    edad = int(np.clip(rng.normal(edad_m, edad_sd), 18, 80))
    satisf = int(np.clip(round(rng.normal(satisf_m, 0.7)), 1, 5))
    antig = int(np.clip(rng.gamma(2.0, antig_m / 2.0), 1, 180))

    filas.append({
        "id_cliente": f"CL{i:05d}",
        "genero": "F" if rng.random() < p_fem else "M",
        "edad": edad,
        "sucursal": rng.choice(SUCURSALES, p=[0.34, 0.26, 0.22, 0.18]),
        "canal_preferido": "En línea" if rng.random() < PROB_ONLINE[perfil] else "Tienda física",
        "tarjeta_departamental": "Sí" if rng.random() < p_tarjeta else "No",
        "antiguedad_meses": antig,
        "compras_anuales": compras,
        "gasto_anual": round(gasto, 2),
        "ticket_promedio": round(gasto / compras, 2),
        "dias_ultima_compra": recencia,
        "pct_compras_descuento": round(descuento * 100, 1),
        "familia_olfativa": rng.choice(FAMILIAS, p=PROB_FAMILIA[perfil]),
        "concentracion_preferida": rng.choice(CONCENTRACIONES, p=PROB_CONC[perfil]),
        "linea_preferida": rng.choice(["Diseñador", "Nicho", "Celebridad / masivo"],
                                      p=PROB_LINEA[perfil]),
        "satisfaccion": satisf,
    })

df = pd.DataFrame(filas)

# ------------------------------------------------------------------
# Problemas de calidad introducidos a propósito
# ------------------------------------------------------------------
# 1. Género capturado de distintas formas
variantes = {"F": ["F", "Femenino", "mujer", " f ", "FEMENINO"],
             "M": ["M", "Masculino", "hombre", " m", "MASCULINO"]}
idx = rng.choice(df.index, size=160, replace=False)
df.loc[idx, "genero"] = [rng.choice(variantes[g]) for g in df.loc[idx, "genero"]]

# 2. Sucursal con mayúsculas y espacios inconsistentes
idx = rng.choice(df.index, size=70, replace=False)
df.loc[idx, "sucursal"] = df.loc[idx, "sucursal"].map(
    lambda s: rng.choice([s.upper(), s.lower(), f" {s} "]))

# 3. Edades imposibles (errores de captura)
idx = rng.choice(df.index, size=8, replace=False)
df.loc[idx, "edad"] = rng.choice([0, 3, 150, 199, 999], size=8)

# 4. Recencia negativa (fecha de última compra mal capturada)
idx = rng.choice(df.index, size=6, replace=False)
df.loc[idx, "dias_ultima_compra"] = -df.loc[idx, "dias_ultima_compra"]

# 5. Valores nulos
for col, n in [("edad", 25), ("genero", 18), ("familia_olfativa", 22),
               ("satisfaccion", 30), ("canal_preferido", 12)]:
    idx = rng.choice(df.index, size=n, replace=False)
    df.loc[idx, col] = np.nan

# 6. Gasto anual guardado como texto en algunas filas ("$12,345.67")
df["gasto_anual"] = df["gasto_anual"].astype(object)
idx = rng.choice(df.index, size=90, replace=False)
df.loc[idx, "gasto_anual"] = df.loc[idx, "gasto_anual"].map(lambda v: f"${v:,.2f}")

# 7. Registros duplicados (el mismo cliente exportado dos veces)
duplicados = df.sample(n=20, random_state=SEMILLA)
df = pd.concat([df, duplicados]).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)

df.to_csv(SALIDA, index=False, encoding="utf-8")
print(f"Dataset generado: {SALIDA.relative_to(RAIZ)}")
print(f"Registros: {len(df)} | Columnas: {df.shape[1]}")
