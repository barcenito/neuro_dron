# Índice

- `src/drone_swarm/direct/`: piloto directo, entrenamiento, escenarios y modelos.
- `src/drone_swarm/learning/`: red neuronal y evolución de parámetros.
- `src/drone_swarm/simulation/`: física, geometría y visor compartidos.
- `src/drone_swarm/perception/`: sensores y funciones de percepción compartidas.
- `src/drone_swarm/control/`: funciones compartidas requeridas por la simulación.
- `src/drone_swarm/artifacts/`: utilidades necesarias para exportar modelos.
- `src/drone_swarm/config.py`: configuración compartida.
- `src/drone_swarm/runtime.py`: selección de CPU/GPU.
- `tools/train_direct.py`: entrenamiento del piloto.
- `tools/eval_direct.py`: evaluación, visualización y exportación.
- `tools/test_components.py`: ejecución de pruebas de componentes.
- `tests/integration/swarm_v2/`: pruebas del piloto directo; se conserva la ruta interna.
- `pyproject.toml`: dependencias e instalación del paquete.


- `tools/train_safe.py`: entrenamiento con límites de RAM/CPU y exclusión de ejecuciones simultáneas (Linux con systemd).
