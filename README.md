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

La tolerancia de clusters tambien se interpreta como una distancia del dominio
base `[0, 1]`: `cluster_tolerance=0.001` equivale a una tolerancia efectiva
`0.002` en `[-1, 1]` y en la variante `unbounded`.

El ATBCR se valida con la condicion estricta `epsilon < theta`.

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

- [models](src/atbcr_analysis/models): reglas de interaccion, contratos y factory de modelos.
- [graphs](src/atbcr_analysis/graphs): generadores de grafos, contratos y factory de grafos.
- [simulation.py](src/atbcr_analysis/simulation.py): motor generico de simulacion.
- [normalizers.py](src/atbcr_analysis/normalizers.py): politicas de normalizacion independientes del modelo.
- [metrics.py](src/atbcr_analysis/metrics.py): metricas por snapshot.
- [config.py](src/atbcr_analysis/config.py): configuracion de escenarios, grafos, modelos, metricas y variantes.
- [persistence.py](src/atbcr_analysis/persistence.py): escritura de resultados.
- [analysis.py](src/atbcr_analysis/analysis.py): comparacion de distribuciones finales.
- [plots.py](src/atbcr_analysis/plots.py): graficos desde CSV persistidos.

Las clases de [config.py](src/atbcr_analysis/config.py)
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

- `unbounded_extreme_cutoff`
- `cluster_tolerance`
- `record_every`
- `store_opinion_snapshots`
- `store_interaction_events`

La extremizacion se define de forma explicita por dominio:

- `bounded_01`: una opinion es extrema si `x <= 0.1` o `x >= 0.9`.
- `bounded_m11`: una opinion es extrema si `abs(x) >= 0.8`, que es la imagen
  exacta de la regla anterior bajo `y = 2x - 1`.
- `unbounded`: se usa `unbounded_extreme_cutoff` como corte configurable sobre
  `abs(x)`, porque no hay extremos naturales del intervalo.

`extremized_threshold` se mantiene temporalmente como alias de compatibilidad
para `unbounded_extreme_cutoff`, pero solo afecta a la variante `unbounded`.

`cluster_count` usa una regla 1D basada en ancla: se ordenan las opiniones, la
primera opinion de un cluster queda como `anchor`, y las siguientes entran en
ese cluster solo si estan a distancia menor o igual que la tolerancia efectiva
respecto a ese `anchor`. Si no, abren un cluster nuevo y pasan a ser el nuevo
`anchor`. No es una regla basada en gaps consecutivos.

## Metricas registradas

Cada snapshot guarda:

- frecuencia acumulada de confianza;
- frecuencia acumulada de inaccion;
- frecuencia acumulada de repulsion;
- frecuencia local por ventana de confianza, inaccion y repulsion, calculada
  entre snapshots;
- `max(abs(opinions))`;
- `mean(abs(opinions))`;
- desviacion estandar;
- percentiles `p50`, `p90`, `p95` y `p99` de `abs(opinions)`;
- porcentaje de agentes extremizados;
- numero de clusters;
- tolerancia de clusters configurada y tolerancia efectiva;
- opiniones por agente, salvo que se use `--no-snapshots`.

Las agregaciones Monte Carlo de `summary.csv` incluyen `mean`, `std`, `median` e
IC95% de la media para las metricas finales principales, incluidas las
frecuencias por ventana. El IC95% usa la aproximacion normal
`mean +/- 1.96 * sample_std / sqrt(n)`. Con `runs = 1`, el IC95% se escribe como
`NaN` porque la incertidumbre no es estimable con una sola replica.

## Reproducibilidad

Cada run conserva una `seed` maestra y tres seeds efectivas:

- `graph_seed`
- `opinion_seed`
- `dynamics_seed`

Si no se pasan explicitamente, se derivan de la seed maestra. Para comparaciones
pareadas entre variantes, usa la misma `seed` o las mismas tres seeds explicitas
en todas las variantes. El CLI permite fijarlas con:

```powershell
atbcr --seed 7 --graph-seed 100 --opinion-seed 200 --dynamics-seed 300
```

## Instalacion local

```powershell
py -m pip install -e .
```

## Ejecucion base

```powershell
py -m atbcr_analysis.cli --scenarios repulsivo:0.2:0.7:0.1,intermedio:0.3:0.8:0.1,confianza:0.4:0.9:0.1 --output-dir runs/article-scenarios
```

Tambien se puede lanzar la misma configuracion desde YAML. Los nombres de las
claves coinciden con los flags del CLI, pero usando guion bajo en vez de guion:

```yaml
scenarios:
  - repulsivo:0.2:0.7:0.1
  - intermedio:0.3:0.8:0.1
  - confianza:0.4:0.9:0.1
variants:
  - baseline_01
  - bounded_m11
  - unbounded
steps: 13500
runs: 20
workers: max
seed: 7
record_every: 500
output_dir: runs/article-scenarios
```

```powershell
atbcr --config experiments.yaml
```

Para lanzar varias pruebas seguidas, usa `defaults` y `experiments`:

```yaml
defaults:
  variants: [unbounded]
  steps: 13500
  runs: 20
  record_every: 500

experiments:
  - scenarios: [repulsivo:0.2:0.7:0.1]
    normalizer: max_abs
    normalization_every: [100, 500, 1000]
    output_dir: runs/max-abs-repulsivo
  - scenarios: [repulsivo:0.2:0.7:0.1]
    normalizer: signed_log_max_abs
    normalization_every: [100, 500, 1000]
    output_dir: runs/signed-log-repulsivo
```

Los flags del CLI sobrescriben el YAML, por ejemplo `atbcr --config
experiments.yaml --seed 99`.

Los YAML tambien aceptan `scenario_grid` para construir barridos reproducibles
sin enumerar cada escenario manualmente:

```yaml
scenario_grid:
  epsilon: {start: 0, stop: 1, step: 0.05}
  theta: {start: 0, stop: 1, step: 0.05}
  epsilon_less_than_theta: true
  mu: 0.1
```

Hay presets listos para ejecutar en:

- [experiments/article_figure_3.yaml](experiments/article_figure_3.yaml): producto cruzado
  `epsilon = [0.2, 0.3, 0.4]` y `theta = [0.7, 0.8, 0.9]`.
- [experiments/article_figure_3_diagonal.yaml](experiments/article_figure_3_diagonal.yaml):
  tres casos diagonales para validaciones rapidas.
- [experiments/article_sensitivity.yaml](experiments/article_sensitivity.yaml): barrido
  `epsilon, theta` con paso `0.05` y condicion `epsilon < theta`.

`workers` acepta un numero concreto o `max`/`auto` para usar todos los nucleos
disponibles:

```yaml
workers: max
```

Durante la ejecucion, `atbcr` muestra una barra por cada combinacion
`scenario/variant/normalizer`, con las replicas completadas y cuantas quedan.
Se puede desactivar con `--no-progress`:

```powershell
atbcr --config experiments.yaml --no-progress
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

Cuando `--normalizer` no es `none`, `--normalization-every` es obligatorio y
debe ser positivo. Esto evita ejecutar silenciosamente un experimento sin
normalizacion por una configuracion incompleta.

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

- `summary.csv`: estadistica descriptiva e IC95% de metricas finales por
  escenario, variante y normalizacion.
- `trajectories.csv`: metricas temporales por replica, incluyendo frecuencias
  acumuladas, frecuencias por ventana, percentiles y seeds efectivas.
- `snapshots.csv`: opiniones temporales por agente, con seeds efectivas.
- `normalization_events.csv`: evento por normalizacion aplicada, con
  `max_abs_before`, `max_abs_after` y `scale` cuando aplica.
- `interaction_events.csv`: eventos de interaccion si se activa
  `--store-interaction-events`. Por defecto esta desactivado y no guarda
  eventos `inaction`, porque no cambian opiniones y multiplican el volumen.
- `config.json`: configuracion completa de la ejecucion, incluyendo escenarios,
  variantes, normalizadores y frecuencias.

## Comparacion de distribuciones

El modulo `analysis.py` incluye distancia Wasserstein 1D empirica. Por defecto
`atbcr-compare` compara variantes de forma pareada por seed: calcula una
distancia por seed y resume `mean`, `std`, `median` e IC95%. Exige las mismas
`run_seed`, `graph_seed`, `opinion_seed`, `dynamics_seed` y el mismo numero de
agentes por seed. Ademas de `wasserstein.csv`, escribe
`wasserstein_by_seed.csv` con las distancias individuales. La comparacion
agregada anterior, que mezcla todas las opiniones finales antes de comparar,
sigue disponible con `--aggregate`.

```powershell
atbcr-compare runs/reference/snapshots.csv --output runs/reference/wasserstein.csv
atbcr-compare runs/reference/snapshots.csv --aggregate --output runs/reference/wasserstein-aggregate.csv
```

## Benchmark

Para medir una aproximacion de rendimiento antes de lanzar campanas largas:

```powershell
py scripts/benchmark.py --agents 1000 --steps 100000 --runs 1 10 100
```

El script imprime segundos totales, segundos por run e interacciones por
segundo. No contiene resultados hardcodeados.

## Plots

El comando `atbcr-plot` genera graficos a partir de los CSV persistidos por
`atbcr`. Tiene tres subcomandos:

- `metric`: lee `trajectories.csv` y dibuja una metrica temporal por serie
  `scenario/variant/normalizer/normalization_every`. Por defecto usa
  `max_abs_opinion`, pero se puede cambiar con `--metric`.
- `distribution`: lee `snapshots.csv`, selecciona el ultimo `step` de cada
  replica y dibuja histogramas de opiniones finales. Por defecto usa 50 bins,
  configurable con `--bins`.
- `opinions`: lee `snapshots.csv` y dibuja como cambian las opiniones de los
  agentes a lo largo del tiempo para una replica concreta. Se puede filtrar por
  `--scenario`, `--variant`, `--normalizer`, `--normalization-every` y
  `--run-seed`.

Ambos subcomandos requieren `--output` con la ruta del archivo de imagen. La
extension elegida determina el formato que escribira Matplotlib, por ejemplo
`.png`, `.pdf` o `.svg`.

```powershell
atbcr-plot metric runs/reference/trajectories.csv --metric max_abs_opinion --output runs/reference/max_abs.png
atbcr-plot distribution runs/reference/snapshots.csv --bins 50 --output runs/reference/final_distribution.png
atbcr-plot opinions runs/reference/snapshots.csv --scenario repulsivo --variant unbounded --normalizer max_abs --normalization-every 100 --run-seed 7 --output runs/reference/opinions.png
```
