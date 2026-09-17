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
