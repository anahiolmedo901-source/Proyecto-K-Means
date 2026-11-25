# Segmentación de clientes del departamento de perfumería mediante K-Means

Proyecto integrador de minería de datos. Se aplica el algoritmo de aprendizaje no supervisado **K-Means** a 1,500 clientes del departamento de perfumería de una tienda departamental para identificar segmentos según su comportamiento de compra (recencia, frecuencia, gasto y uso de descuentos). Los resultados se presentan en un **dashboard interactivo** hecho con Streamlit.

**Integrante:** Viviana Anahí Olmedo García  
**Docente:** Mtra. Julia Elizabeth García Herrera  
**Grupo:** IDGS 9-2  
**Fecha:** 23/Noviembre/2026

## Resultados principales

- Se evaluaron valores de K entre 2 y 8. El mejor fue **K = 4** (silueta = 0.407, Davies-Bouldin = 0.927). El modelo es estable: con 10 semillas distintas, el ARI promedio es de 0.984.
- **Clientes premium** (277 clientes, 18%): compran 9.9 veces al año, gastan $25,404 en promedio y casi no usan descuentos. Generan el **56% de las ventas**.
- **Cazadores de ofertas** (434 clientes, 29%): compran seguido, pero el 71% de sus compras lleva descuento y su ticket es el más bajo ($943).
- **Compradores ocasionales** (478 clientes, 32%): compran de 2 a 4 veces al año, con ticket medio; probablemente compran para regalo.
- **Clientes inactivos** (311 clientes, 21%): llevan 250 días en promedio sin comprar y aportan solo el 4% de las ventas.

## Dataset

- **Nombre:** Clientes del departamento de perfumería (`clientes_perfumeria.csv`).
- **Fuente:** dataset **propio simulado**, generado con `src/generar_dataset.py` (semilla fija 2026). No hay un dataset público de ventas de perfumería de una tienda departamental, así que los datos se crearon con reglas de negocio razonables. No pertenecen a ninguna empresa real.
- **Tamaño:** 1,520 registros (1,500 clientes y 20 duplicados) y 16 columnas.
- **Calidad:** incluye a propósito errores reales de captura (nulos, duplicados, texto inconsistente, edades imposibles y montos guardados como texto) para practicar la limpieza.
- **Variables del modelo:** días desde la última compra, compras al año, gasto anual y % de compras con descuento.

## Herramientas utilizadas

- Python 3.10 o superior
- Pandas y NumPy: generación, carga, limpieza y análisis de datos
- Scikit-learn: StandardScaler, KMeans y métricas de evaluación
- Matplotlib y Seaborn: gráficas
- Streamlit: dashboard interactivo

## Estructura del repositorio

```
proyecto_kmeans/
├── data/
│   ├── original/clientes_perfumeria.csv              # dataset original (con errores)
│   └── procesado/clientes_perfumeria_clusters.csv    # dataset limpio con el clúster asignado
├── src/
│   ├── generar_dataset.py                            # genera el dataset simulado
│   └── analisis_kmeans.py                            # exploración, limpieza, K-Means e interpretación
├── dashboard/
│   └── app.py                                        # dashboard de Streamlit (2 páginas)
├── .streamlit/config.toml                            # tema del dashboard (colores)
├── graficas/                                         # gráficas generadas por el análisis
├── resultados/                                       # métricas, centroides, tablas y salida de consola
├── docs/
│   ├── Documentacion_Proyecto_KMeans.pdf             # documentación del proyecto
│   ├── Documentacion_Proyecto_KMeans.docx            # versión editable
│   ├── generar_documento.js                          # script que genera la documentación
│   └── img/                                          # imágenes usadas en el documento
├── requirements.txt
└── README.md
```

## Cómo ejecutar el proyecto

1. Clona el repositorio y entra a la carpeta:
   ```bash
   git clone https://github.com/anahiolmedo901-source/Proyecto-K-Means.git
   cd Proyecto-K-Means
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. (Opcional) Vuelve a generar el dataset original. Con la misma semilla siempre se obtiene el mismo archivo:
   ```bash
   python src/generar_dataset.py
   ```
4. Ejecuta el análisis. El script imprime cada parte en la consola y vuelve a generar las gráficas, los resultados y el dataset procesado:
   ```bash
   python src/analisis_kmeans.py
   ```
5. Abre el dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
   Se abrirá en el navegador en `http://localhost:8501`.

## Dashboard

- **Página 1. Análisis general:** 4 KPI (clientes, ventas anuales, ticket promedio y % de compras con descuento), histograma de cualquier variable, clientes por categoría, matriz de correlación, ventas por sucursal y canal, estadísticas descriptivas y resumen de la limpieza.
- **Página 2. Segmentación K-Means:** métricas del modelo, distribución por segmento, dispersión con los centros de cada grupo, clientes contra ventas, comparación de una variable por segmento, perfil de los segmentos, preferencias por segmento, estadísticas por segmento, selección de K, comparación de configuraciones, descripción de los grupos, hallazgos y recomendaciones.
- **Filtros (barra lateral):** segmento, sucursal, género y rango de edad.
- **Diseño:** fondo crema, paleta pastel validada para daltonismo (lavanda, rosa, miel y salvia; un color fijo por segmento) y tipografía Fraunces + DM Sans.

> Los datos son simulados: los resultados ilustran la metodología y no describen a una tienda real.
