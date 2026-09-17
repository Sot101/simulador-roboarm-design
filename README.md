# miniLab — Reachable workspace viewer

Visualizador 3D interactivo (matplotlib standalone, GUI diseñada para 1366x768) del espacio de trabajo
alcanzable de un brazo serial de 1–6 DOF, calculado por Monte Carlo sobre
cinemática directa DH estándar. Ver docstring de `workspace_viewer.py` para
detalles y limitaciones.

## Ejecutar

```bash
# el .venv ya está creado con uv (Python 3.12 gestionado, incluye tkinter)
.venv/bin/python workspace_viewer.py
```

Si hay que recrearlo (el python3 del sistema no trae `venv` ni `tkinter`):

```bash
uv venv --managed-python --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

## Controles

| Widget | Acción |
|---|---|
| RadioButtons **DOF** | Cambia el preset 1–6 DOF (resetea longitudes y pose) |
| Sliders **L1..L6** | Longitud de cada eslabón (parámetro `a` o `d` del DH, indicado en la etiqueta). Solo se muestran los del DOF activo. Mover uno marca la nube como desactualizada |
| Slider **N muestras** | Tamaño del Monte Carlo (500–20000) |
| **Recalcular workspace** | Lanza el muestreo (manual, no en cada arrastre) |
| **Pose aleatoria** / **Pose home** | Cambia la pose en la que se dibuja el brazo |
| **Reset longitudes** | Restaura las longitudes del preset |
| Checkboxes | Muestra/oculta brazo y nube de puntos (independientes) |

## Versión web (Three.js, sin servidor)

- `workspace_lab.html` — fuente publicada como artefacto en claude.ai (misma cinemática y presets que la versión Python).
- `workspace_lab_local.html` — misma página envuelta en `<!doctype>`/`<head>` para abrirla directamente en el navegador (`xdg-open workspace_lab_local.html`). Necesita internet solo para descargar Three.js y las fuentes desde CDN.

## Cinemática inversa y trayectorias (versión web)

- **Seleccionar objetivo**: clic sobre cualquier punto de la nube (o «Elegir uno al azar»). Cada punto guarda la configuración articular que lo generó durante el muestreo, así que siempre existe al menos una solución conocida.
- **IK numérica de posición**: Jacobiano geométrico + mínimos cuadrados amortiguados (DLS), saturación en los límites articulares y término de espacio nulo hacia *home* para brazos redundantes. Multistart: primero desde *home*; si cae en un mínimo local, hasta 12 semillas aleatorias (se elige la más cercana a *home*); último recurso, la configuración del muestreo. El panel muestra θ IK y θ muestra lado a lado.
- **Ruta directa**: línea recta del TCP dividida en 48 waypoints, IK encadenada (cada waypoint parte de la solución anterior). Si un waypoint no es alcanzable o hay un salto articular > 45° (cambio de rama), se informa y se cae a interpolación articular.
- **Por articulación**: desde *home*, J1 → θ1, luego J2 → θ2, … una junta cada vez, con pausa entre ellas.
- Animación con velocidad ajustable, línea de la ruta planificada, estela del TCP, barra de progreso y error final del TCP en mm. «Volver a home» interpola en espacio articular.
