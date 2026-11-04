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
- [config](src/atbcr_analysis/config): schemas de configuracion de escenarios,
  grafos, modelos, metricas, variantes y archivos YAML.
- [persistence.py](src/atbcr_analysis/persistence.py): escritura de resultados.
- [analysis.py](src/atbcr_analysis/analysis/analysis.py): comparacion de distribuciones finales.
- [plots.py](src/atbcr_analysis/plots/plots.py): graficos desde CSV persistidos.

Las clases de [config](src/atbcr_analysis/config) usan Pydantic para validar
la configuracion en la frontera del sistema. El
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

Para `CompleteGraphConfig`, la adjacency y la secuencia de aristas se representan
de forma lazy. El ATBCR no necesita inspeccionar vecinos para actualizar un par,
asi que una simulacion con `n_agents=1000` no materializa las relaciones O(N2)
del grafo completo solo para seleccionar interacciones aleatorias.

La configuracion de metricas se agrupa en `MetricsConfig`:

- `unbounded_extreme_cutoff`
- `cluster_tolerance`
- `record_every`
- `store_opinion_snapshots`
- `store_normalization_events` (desactivalo en barridos largos con intervalos
  pequenos para evitar retener un evento por normalizacion y replica)
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

Tambien se puede lanzar la misma configuracion desde YAML. El fichero tiene un
schema propio y no reutiliza la sintaxis compacta de los flags del CLI:

```yaml
experiments:
  - scenarios:
      - {name: repulsivo, epsilon: 0.2, theta: 0.7, mu: 0.1}
      - {name: intermedio, epsilon: 0.3, theta: 0.8, mu: 0.1}
      - {name: confianza, epsilon: 0.4, theta: 0.9, mu: 0.1}
    variants:
      - name: baseline_01
        domain: bounded_01
        initializer: {kind: uniform, low: 0.0, high: 1.0}
      - name: bounded_m11
        domain: bounded_m11
        threshold_scale: 2.0
        initializer: {kind: uniform, low: -1.0, high: 1.0}
      - name: unbounded
        domain: unbounded
        threshold_scale: 2.0
        initializer: {kind: uniform, low: -1.0, high: 1.0}
    graph: {kind: complete, n_agents: 1000}
    steps: 13500
    metrics: {record_every: 500}
    monte_carlo: {runs: 20, workers: max, seed: 7}
    output_dir: runs/article-scenarios
```

```powershell
atbcr --config experiments.yaml
```

Para lanzar varias pruebas seguidas, usa `defaults` y `experiments`:

```yaml
defaults:
  variants:
    - name: unbounded
      domain: unbounded
      threshold_scale: 2.0
      initializer: {kind: uniform, low: -1.0, high: 1.0}
  steps: 13500
  metrics: {record_every: 500}
  monte_carlo: {runs: 20}

experiments:
  - scenarios:
      - {name: repulsivo, epsilon: 0.2, theta: 0.7, mu: 0.1}
    normalizations:
      - {kind: max_abs, every: 100}
      - {kind: max_abs, every: 500}
      - {kind: max_abs, every: 1000}
    output_dir: runs/max-abs-repulsivo
  - scenarios:
      - {name: repulsivo, epsilon: 0.2, theta: 0.7, mu: 0.1}
    normalizations:
      - {kind: signed_log_max_abs, every: 100}
      - {kind: signed_log_max_abs, every: 500}
      - {kind: signed_log_max_abs, every: 1000}
    output_dir: runs/signed-log-repulsivo
```

Los YAML tambien aceptan `scenario_grid` para construir barridos reproducibles
sin enumerar cada escenario manualmente:

```yaml
experiments:
  - scenario_grid:
      epsilon: {start: 0, stop: 1, step: 0.05}
      theta: {start: 0, stop: 1, step: 0.05}
      epsilon_less_than_theta: true
      mu: 0.1
    variants:
      - name: baseline_01
        domain: bounded_01
        initializer: {kind: uniform, low: 0.0, high: 1.0}
    graph: {kind: complete, n_agents: 1000}
    steps: 100000
    metrics: {record_every: 1000}
    monte_carlo: {runs: 100, workers: max, seed: 7}
    output_dir: runs/article-sensitivity
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

## Umbrales dinamicos

El experimento 0.7 mantiene las opiniones sin clipping ni normalizacion y adapta
los umbrales a la escala relativa de cada ejecucion. La variante usa
`threshold_mode: max_abs`; `threshold_update_every` indica cuantos pasos se
completan entre actualizaciones. El primer intervalo emplea los umbrales ya
transformados al dominio de la variante. En cada actualizacion posterior se usa
el cociente entre el maximo absoluto actual y el maximo absoluto inicial.

`threshold_scale: 2.0` transforma una sola vez los umbrales de `[0,1]` a
`[-1,1]`. El escalado dinamico se aplica despues sobre esos valores efectivos.
Si el maximo inicial es cero se conservan los umbrales iniciales. Un valor no
finito en las opiniones, en la escala o en los umbrales detiene la ejecucion con
un `FloatingPointError` que identifica el paso; no se aplica ningun limite
artificial.

```powershell
atbcr --config experiments/experiment_0.7.yaml
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

- `summary.csv`: estadistica descriptiva e IC95% de metricas finales por
  escenario, variante y normalizacion.
- `trajectories.csv`: metricas temporales por replica, incluyendo frecuencias
  acumuladas, frecuencias por ventana, percentiles y seeds efectivas.
- `snapshots.csv`: opiniones temporales por agente, con seeds efectivas.
- `final_opinions.csv`: solo las opiniones del ultimo snapshot de cada replica;
  es la entrada recomendada para histogramas y comparaciones de distribuciones.
- `normalization_events.csv`: evento por normalizacion aplicada, con
  `max_abs_before`, `max_abs_after` y `scale` cuando aplica.
- `interaction_events.csv`: eventos de interaccion si se activa
  `--store-interaction-events`. Por defecto esta desactivado y no guarda
  eventos `inaction`, porque no cambian opiniones y multiplican el volumen.
- `config.json`: configuracion completa de la ejecucion, incluyendo escenarios,
  variantes, normalizadores y frecuencias.

## Comparacion de distribuciones

El modulo `analysis.py` incluye distancia Wasserstein 1D empirica. Por defecto
`atbcr-compare` compara solo series cientificamente compatibles: mismo
`scenario`, mismo dominio de opiniones y distinta combinacion
`variant/normalizer/normalization_every`. Esto evita comparar silenciosamente
distribuciones en `[0, 1]` contra distribuciones en `[-1, 1]`.

El caso principal queda directo: comparar `unbounded/none` contra
`unbounded/max_abs/k` y `unbounded/signed_log_max_abs/k` dentro del mismo
escenario. La comparacion es pareada por seed: calcula una distancia por seed y
resume `mean`, `std`, `median` e IC95%. Exige las mismas `run_seed`,
`graph_seed`, `opinion_seed`, `dynamics_seed` y el mismo numero de agentes por
seed. Ademas de `wasserstein.csv`, escribe `wasserstein_by_seed.csv` con las
distancias individuales.

La comparacion agregada anterior, que mezcla todas las opiniones finales antes
de comparar, sigue disponible con `--aggregate`. Las comparaciones entre
escenarios distintos o dominios distintos requieren opt-in explicito:
`--cross-scenario` y `--allow-cross-domain`.

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
`atbcr`. Tiene nueve subcomandos:

- `metric`: lee `trajectories.csv` y dibuja una metrica temporal por serie
  `scenario/variant/normalizer/normalization_every`. Por defecto usa
  `max_abs_opinion`, pero se puede cambiar con `--metric`.
- `compare`: compara de forma generica metricas temporales procedentes de uno o
  varios `trajectories.csv`. Permite filtrar configuraciones, transformar cada
  replica antes de agregar, mostrar IC95% y ajustar rectas por replica.
- `distribution`: lee `snapshots.csv`, selecciona el ultimo `step` de cada
  replica y dibuja histogramas de opiniones finales. Por defecto usa 50 bins,
  configurable con `--bins`.
- `final-opinions`: reduce un `snapshots.csv` existente a un
  `final_opinions.csv` con el ultimo `step` de cada replica. La salida se puede
  cambiar con `--output`.
- `opinions`: lee `snapshots.csv` y dibuja como cambian las opiniones de los
  agentes a lo largo del tiempo para una replica concreta. Se puede filtrar por
  `--scenario`, `--variant`, `--normalizer`, `--normalization-every` y
  `--run-seed`. `--yscale symlog` permite representar simultaneamente valores
  pequenos, cero y opiniones positivas o negativas de gran magnitud.
- `opinion-metrics`: deriva, desde `final_opinions.csv` o `snapshots.csv`, la
  proporcion de agentes con `|x_i| < 1`, `min_i |x_i|` y
  `min_{i != j} |x_i - x_j|`. Escribe un CSV compacto y permite filtrar por
  `--epsilon`, `--theta`, variante, normalizador, frecuencia y semilla.
- `extremized-heatmap`: lee `summary.csv` y representa
  `extremized_share_mean` para cada combinacion de epsilon y theta. Acepta
  columnas `epsilon` y `theta`, o nombres de escenario generados por la rejilla
  (`e0p1_t0p5`). Se puede filtrar por `--variant`, `--normalizer` y
  `--normalization-every`.
- `parameter-distribution`: lee `final_opinions.csv` (o `snapshots.csv`) y
  genera el histograma de opiniones finales. Acepta `--epsilon` y `--theta`, o
  varios argumentos `--pair EPSILON THETA` para comparar las distribuciones en
  subplots con bins y escalas compartidos. El rango de opinion se obtiene del
  dominio de la variante; para dominios no acotados se usan los datos observados.
  `--run-seed` limita las distribuciones a una sola ejecucion Monte Carlo.
- `extremized-trajectory`: lee `trajectories.csv` y representa la media entre
  replicas de `extremized_share(t)`. Acepta `--epsilon` y `--theta` para una
  curva, o varios argumentos `--pair EPSILON THETA` para comparar curvas con
  leyenda en una misma figura.
- `stability`: lee uno o varios `snapshots.csv` y calcula, para cada ejecucion e
  intervalo entre snapshots consecutivos, el maximo cambio absoluto de opinion
  entre agentes. Permite seleccionar una replica con `--run-seed`, procesa el
  archivo por bloques y puede crear un CSV compacto reutilizable.

Todos los subcomandos que generan figuras requieren `--output` con la ruta del
archivo de imagen. La extension elegida determina el formato que escribira
Matplotlib, por ejemplo `.png`, `.pdf` o `.svg`.

```powershell
atbcr-plot metric runs/reference/trajectories.csv --metric max_abs_opinion --output runs/reference/max_abs.png
atbcr-plot compare --input runs/experiment_0.1/trajectories.csv runs/experiment_0.3/trajectories.csv --variants baseline_01 unbounded --metric max_abs_opinion --aggregate mean --ci --output runs/comparison/max_abs.pdf
atbcr-plot compare --input runs/experiment_0.3/trajectories.csv --pair 0.2 0.7 --variants unbounded --normalizers none --metric max_abs_opinion --transform log --fit --fit-start 1000 --fit-end 10000 --output runs/comparison/log_growth.svg
atbcr-plot distribution runs/reference/snapshots.csv --bins 50 --output runs/reference/final_distribution.png
atbcr-plot final-opinions runs/reference/snapshots.csv
atbcr-plot opinions runs/reference/snapshots.csv --scenario repulsivo --variant unbounded --normalizer max_abs --normalization-every 100 --run-seed 7 --output runs/reference/opinions.png
atbcr-plot opinions runs/experiment_0.3/snapshots.csv --scenario e0p2_t0p7 --variant unbounded --normalizer none --run-seed 7 --yscale symlog --symlog-linthresh 1 --output runs/experiment_0.3/opinions_e0p2_t0p7_seed7_symlog.png
atbcr-plot opinion-metrics runs/experiment_0.3/final_opinions.csv --epsilon 0.2 --theta 0.7 --variant unbounded --normalizer none --run-seed 7 --output runs/experiment_0.3/opinion_metrics_e0p2_t0p7_seed7.csv
atbcr-plot extremized-heatmap runs/grid/summary.csv --variant unbounded --normalizer none --output runs/grid/extremized.png
atbcr-plot parameter-distribution runs/experiment_0.1/final_opinions.csv --epsilon 0.2 --theta 0.7 --variant baseline_01 --normalizer none --output runs/experiment_0.1/distribution_e0p2_t0p7.png
atbcr-plot parameter-distribution runs/experiment_0.1/final_opinions.csv --pair 0.2 0.7 --pair 0.3 0.8 --pair 0.4 0.9 --variant baseline_01 --normalizer none --output runs/experiment_0.1/distributions_comparison.png
atbcr-plot parameter-distribution runs/experiment_0.4/final_opinions.csv --pair 0.2 0.7 --pair 0.3 0.8 --variant unbounded --normalizer max_abs --normalization-every 10 --run-seed 7 --output runs/experiment_0.4/distributions_seed_7.png
atbcr-plot extremized-trajectory runs/experiment_0.3/trajectories.csv --epsilon 0.2 --theta 0.7 --variant unbounded --normalizer none --output runs/experiment_0.3/extremized_e0p2_t0p7.png
atbcr-plot extremized-trajectory runs/experiment_0.3/trajectories.csv --pair 0.2 0.7 --pair 0.3 0.8 --pair 0.4 0.9 --variant unbounded --normalizer none --output runs/experiment_0.3/extremized_comparison.png
atbcr-plot stability --input runs/experiment_0.4/snapshots.csv --scenarios e0p2_t0p7 --variants unbounded --normalizers max_abs --normalization-every 10 100 1000 10000 --run-seed 7 --cache runs/experiment_0.4/stability_run7.csv --output runs/experiment_0.4/stability_run7.png
```

### Ejecutar varias graficas en paralelo

Para generar varias graficas independientes a la vez, copia
`scripts/plot_commands.example.txt`, deja un comando completo por linea y
ejecuta:

```powershell
.\.venv\Scripts\python.exe scripts\run_parallel.py scripts\plot_commands.txt --jobs 4
```

`--jobs` limita el numero de procesos simultaneos para no agotar la memoria.
Las lineas vacias y las que empiezan por `#` se ignoran. La salida lleva el
numero del comando como prefijo y el proceso devuelve codigo `1` si falla al
menos uno. Antes de lanzar un lote largo se puede comprobar con `--dry-run`.

Tambien se pueden pasar comandos directamente repitiendo `--command`:

```powershell
.\.venv\Scripts\python.exe scripts\run_parallel.py --jobs 2 `
  --command "atbcr-plot final-opinions runs/experiment_0.1/snapshots.csv" `
  --command "atbcr-plot final-opinions runs/experiment_0.3/snapshots.csv"
```

### Metricas derivadas de opiniones finales o snapshots

`opinion-metrics` genera una fila por replica y `step`, con estas columnas:

- `proportion_abs_lt_1`: proporcion de agentes que permanece en el intervalo
  inicial abierto `(-1, 1)`;
- `closest_to_zero_agent` y `min_abs_opinion`: agente mas proximo a cero y su
  distancia absoluta;
- `closest_pair_agent_i`, `closest_pair_agent_j` y `min_pairwise_distance`:
  pareja de agentes mas proxima y distancia entre sus opiniones;
- `confidence_epsilon`: umbral efectivo de confianza, incluyendo el
  `threshold_scale` de la variante registrado en `config.json`;
- `confidence_pair_exists`: `True` cuando
  `min_pairwise_distance < confidence_epsilon`, la desigualdad estricta usada
  por el modelo ATBCR.

```powershell
atbcr-plot opinion-metrics `
  runs/experiment_0.3/final_opinions.csv `
  --epsilon 0.2 `
  --theta 0.7 `
  --variant unbounded `
  --normalizer none `
  --run-seed 7 `
  --output runs/experiment_0.3/opinion_metrics_e0p2_t0p7_seed7.csv
```

Con `final_opinions.csv` se obtiene una fila final por replica sin recorrer la
historia completa. Si se usa `snapshots.csv`, se obtiene la evolucion de las
metricas para cada `step` almacenado. Con una semilla y filtros exactos, el
lector se detiene al terminar el bloque contiguo seleccionado y solo conserva
un estado en memoria. `snapshot-metrics` se mantiene como alias compatible.
Para grafos no completos, `confidence_pair_exists` indica que existe una pareja
de opiniones suficientemente proxima, pero no garantiza que esa pareja sea una
arista del grafo.

### Estabilidad entre snapshots

`stability` calcula
`Delta_max(t) = max_i |x_i(t) - x_i(t_prev)|`, donde `t_prev` es el snapshot
almacenado anterior de la misma replica. El valor se calcula por ejecucion
Monte Carlo e intervalo exacto. Si no se selecciona `--run-seed`, se representa
la media de las replicas que comparten configuracion e intervalo.

```powershell
atbcr-plot stability `
  --input runs/experiment_0.4/snapshots.csv `
  --scenarios e0p2_t0p7 `
  --variants unbounded `
  --normalizers max_abs `
  --normalization-every 10 100 1000 10000 `
  --run-seed 7 `
  --chunk-size 10000 `
  --cache runs/experiment_0.4/stability_e0p2_t0p7_run7.csv `
  --output runs/experiment_0.4/stability_e0p2_t0p7_run7.pdf
```

`--input` acepta varios archivos. Los filtros `--scenarios`, `--variants`,
`--normalizers` y `--normalization-every` aceptan uno o varios valores. Para
seleccionar la frecuencia vacia del normalizador `none` se puede usar
`--normalization-every none`. Cada punto de la figura corresponde a un cambio
entre dos snapshots almacenados, que pueden estar separados por varios pasos de
simulacion. El lector conserva solo dos snapshots de la ejecucion seleccionada
para emparejar agentes. Cuando todos los filtros y `--run-seed` estan definidos,
busca los bloques exactos mediante un mapa de archivo paginado: no carga los
46 GB en RAM y solo parsea las ejecuciones seleccionadas. El recorrido general
usa bloques acotados por `--chunk-size` como alternativa cuando faltan filtros.

`--cache` guarda un CSV compacto con una fila por ejecucion e intervalo y un
archivo lateral de metadatos. En invocaciones posteriores se reutiliza si el
tamaño y la fecha de los `snapshots.csv`, los filtros y la semilla no han
cambiado. `--refresh-cache` fuerza su reconstruccion. El progreso de lectura se
muestra por bytes; `--no-progress` lo desactiva.

### Comparaciones temporales genericas

`compare` acepta uno o varios valores en `--input`, `--variants`, `--normalizers`
y `--normalization-every`. `--pair EPSILON THETA` se puede repetir para elegir
combinaciones concretas, por ejemplo `--pair 0.2 0.7 --pair 0.3 0.8`. Los
filtros son opcionales: si no se proporciona `--pair`, se representan todos los
pares disponibles. Los parametros se leen de las columnas `epsilon` y `theta`
si existen, del `config.json` situado junto al CSV, o de nombres de rejilla como
`e0p2_t0p7`. Para seleccionar la frecuencia vacia asociada al normalizador
`none`, se puede usar `--normalization-every none`.

Las transformaciones `log` (logaritmo natural) y `log10` se aplican a cada
replica Monte Carlo antes de calcular `mean` o `median`. Un valor cero, negativo
o no finito produce un error con archivo y fila; no se reemplaza por un valor
artificial. Las bandas activadas con `--ci` usan las replicas como unidades
independientes y la aproximacion normal
`agregado +/- 1.96 * desviacion_muestral / sqrt(n)`. Con una sola replica no se
dibuja una banda.

Con `--fit`, el ajuste lineal se realiza por separado sobre cada replica y sobre
la metrica ya transformada. `--fit-start` y `--fit-end` acotan el intervalo. La
leyenda muestra la pendiente y el R2 medios, y se dibuja una recta descriptiva
con los coeficientes medios. Ademas de la figura se crea, por defecto,
`<nombre>_fits.csv`, con resultados por replica y una fila resumen por serie;
`--fit-output` permite elegir otra ruta. El ajuste describe crecimiento en el
intervalo observado y no constituye por si solo una demostracion de divergencia
matematica.

La extension de `--output` determina el formato. Para articulos y memorias se
recomiendan formatos vectoriales (`.pdf` o `.svg`); los formatos raster se
guardan a 300 dpi.
