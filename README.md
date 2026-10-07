# atbcr-analysis

Paquete Python para experimentacion cientifica con modelos de opiniones dinamicas,
con foco inicial en variantes del ATBCR del articulo *Analyzing the extremization
of opinions in a general framework of bounded confidence and repulsion*.

El objetivo principal es comparar el ATBCR original con variantes donde se elimina
o cambia el clipping de opiniones, y estudiar estrategias de normalizacion
periodica sin acoplarlas al modelo.

## Variantes iniciales

- `baseline_01`: ATBCR original con dominio `[0, 1]` y clipping a `[0, 1]`.
- `bounded_m11`: dominio `[-1, 1]` y clipping a `[-1, 1]`.
- `unbounded`: dominio real, sin clipping.

Al transformar el espacio original `[0, 1]` a `[-1, 1]`, las distancias se
duplican. Por eso las variantes `bounded_m11` y `unbounded` aplican:

```text
epsilon' = 2 * epsilon
theta' = 2 * theta
```

## Escenarios

El paquete no trae una campaña privilegiada. Los escenarios se pasan desde
configuracion. Por CLI se definen como:

```text
name:epsilon:theta:mu
```

Por ejemplo, los primeros escenarios del articulo se pueden lanzar como:

| Escenario | epsilon | theta |
| --- | ---: | ---: |
| repulsivo | 0.2 | 0.7 |
| intermedio | 0.3 | 0.8 |
| confianza | 0.4 | 0.9 |

## Arquitectura

- [models](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/models): reglas de interaccion, contratos y factory de modelos.
- [graphs](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/graphs): generadores de grafos, contratos y factory de grafos.
- [simulation.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/simulation.py): motor generico de simulacion.
- [normalizers.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/normalizers.py): politicas de normalizacion independientes del modelo.
- [metrics.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/metrics.py): metricas por snapshot.
- [config.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/config.py): configuracion de escenarios, grafos, modelos, metricas y variantes.
- [persistence.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/persistence.py): escritura de resultados.
- [analysis.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/analysis.py): comparacion de distribuciones finales.
- [plots.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/plots.py): graficos desde CSV persistidos.

Las clases de [config.py](C:/Users/dani_/Documents/GitHub/atbcr-analysis/src/atbcr_analysis/config.py)
usan Pydantic para validar la configuracion en la frontera del sistema. El
motor de simulacion y los resultados siguen usando estructuras ligeras.

La seleccion de modelo se hace con `SimulationConfig`, `ScenarioConfig`,
`ModelConfig` y `ModelFactory`. `ScenarioConfig` describe casos experimentales
de entrada; cada `SimulationConfig` resultante guarda directamente su `name` y
su `model`. Los parametros cientificos viven en la configuracion concreta del
modelo. Para ATBCR eso es `ATBCRModelConfig(epsilon, theta, mu)`. Para anadir
otro modelo no hay que tocar el motor de simulacion: se implementa una clase con
`step(...)` y `validate()`, y se registra con `register_model(...)`.

```python
from atbcr_analysis import ATBCRModelConfig, ScenarioConfig, register_model

register_model("mi_modelo", mi_builder)
scenario = ScenarioConfig(
    name="repulsivo",
    model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
)
```

Los grafos tambien tienen configuraciones especificas por tipo:

- `CompleteGraphConfig(n_agents=1000)`
- `RingGraphConfig(n_agents=1000)`
- `BarabasiAlbertGraphConfig(n_agents=1000, ba_m=3)`
- `ErdosRenyiGraphConfig(n_agents=1000, er_probability=0.02)`
- `WattsStrogatzGraphConfig(n_agents=1000, watts_k=6, watts_rewire_probability=0.1)`

La construccion de grafos sigue el mismo patron que los modelos: `GraphFactory`
usa el campo `kind` de la configuracion y permite registrar generadores nuevos
con `register_graph(...)`.

La configuracion de metricas se agrupa en `MetricsConfig`:

- `extremized_threshold`
- `cluster_tolerance`
- `record_every`
- `store_opinion_snapshots`

## Metricas registradas

Cada snapshot guarda:

- frecuencia acumulada de confianza;
- frecuencia acumulada de inaccion;
- frecuencia acumulada de repulsion;
- `max(abs(opinions))`;
- `mean(abs(opinions))`;
- desviacion estandar;
- porcentaje de agentes extremizados;
- numero de clusters;
- opiniones por agente, salvo que se use `--no-snapshots`.

## Instalacion local

```powershell
py -m pip install -e .
```

## Ejecucion base

```powershell
py -m atbcr_analysis.cli --scenarios repulsivo:0.2:0.7:0.1,intermedio:0.3:0.8:0.1,confianza:0.4:0.9:0.1 --output-dir runs/article-scenarios
```

## Barrido de normalizacion

Normalizacion por `max(abs(x))` cada `k` iteraciones:

```powershell
py -m atbcr_analysis.cli --scenarios repulsivo:0.2:0.7:0.1 --variants unbounded --normalizer max_abs --normalization-every 100,500,1000 --output-dir runs/max-abs-sweep
```

Normalizacion `signed-log` seguida de `max(abs(x))`:

```powershell
py -m atbcr_analysis.cli --scenarios repulsivo:0.2:0.7:0.1 --variants unbounded --normalizer signed_log_max_abs --normalization-every 100,500,1000 --output-dir runs/signed-log-sweep
```

## Inicializacion de opiniones

Uniforme en un rango concreto:

```powershell
py -m atbcr_analysis.cli --initializer uniform --initial-low -1 --initial-high 1
```

Separando preocupados y no preocupados:

```powershell
py -m atbcr_analysis.cli --initializer binary_concern --initial-concern-share 0.3 --non-concern-low -0.2 --non-concern-high 0.4 --concern-low 0.7 --concern-high 1.0
```

## Salidas

Cada ejecucion escribe:

- `summary.csv`: agregados finales por escenario, variante y normalizacion.
- `trajectories.csv`: metricas temporales por replica.
- `snapshots.csv`: opiniones temporales por agente.
- `config.json`: configuracion completa de la ejecucion, incluyendo escenarios,
  variantes, normalizadores y frecuencias.

## Comparacion de distribuciones

El modulo `analysis.py` incluye distancia Wasserstein 1D empirica para comparar
distribuciones finales con semillas pareadas y mismo numero de muestras.

```powershell
atbcr-compare runs/reference/snapshots.csv --output runs/reference/wasserstein.csv
```

## Plots

El comando `atbcr-plot` genera graficos a partir de los CSV persistidos por
`atbcr`. Tiene dos subcomandos:

- `metric`: lee `trajectories.csv` y dibuja una metrica temporal por serie
  `scenario/variant/normalizer/normalization_every`. Por defecto usa
  `max_abs_opinion`, pero se puede cambiar con `--metric`.
- `distribution`: lee `snapshots.csv`, selecciona el ultimo `step` de cada
  replica y dibuja histogramas de opiniones finales. Por defecto usa 50 bins,
  configurable con `--bins`.

Ambos subcomandos requieren `--output` con la ruta del archivo de imagen. La
extension elegida determina el formato que escribira Matplotlib, por ejemplo
`.png`, `.pdf` o `.svg`.

```powershell
atbcr-plot metric runs/reference/trajectories.csv --metric max_abs_opinion --output runs/reference/max_abs.png
atbcr-plot distribution runs/reference/snapshots.csv --bins 50 --output runs/reference/final_distribution.png
```
