# Próximos cambios: predicción y percepción

**Estas mejoras están pendientes: el modelo y el benchmark actuales no incorporan un filtro de Kalman ni sensores nuevos.** El siguiente objetivo es reducir las colisiones con objetos en movimiento y mejorar la transferencia a otros drones, conservando el vuelo individual aprendido.

### Predicción de trayectorias como entrada de la red

Se plantea añadir un módulo de seguimiento basado en **Kalman**, alimentado exclusivamente por las lecturas disponibles en el dron. Sus estimaciones y predicciones se darán como entradas adicionales a la política:

```text
ToF / posibles radares + IMU + estimación del movimiento propio
    → sincronización y compensación de latencia y movimiento propio
    → asociación de mediciones y seguimiento con Kalman
    → posición/velocidad relativas estimadas + incertidumbre
    → predicciones a corto plazo
    → red neuronal → cuatro motores
```

El trabajo previsto es:

1. **Crear una referencia sencilla con los sensores actuales.** Filtrar distancia y velocidad radial de aproximación por sector, teniendo en cuenta ruido, antigüedad y lecturas inválidas. Una variación de distancia también puede deberse al movimiento o giro del propio dron, o a que cambia la superficie observada; no equivale directamente a la velocidad de un obstáculo.
2. **Añadir seguimiento espacial cuando las mediciones lo permitan.** Compensar el movimiento propio y asociar detecciones entre instantes antes de actualizar las pistas. Con detecciones cartesianas puede usarse un Kalman de velocidad constante; con medidas de distancia/ángulo y relaciones no lineales, estudiar un EKF. Una única distancia dentro de un cono no determina por sí sola una posición o trayectoria 3D: hay que mantener esa incertidumbre y evitar inventar información angular.
3. **Definir las nuevas entradas.** Para un número fijo de pistas o sectores: posición y velocidad relativas estimadas, predicción de posición a horizontes como 0,2, 0,5 y 1 s, incertidumbre, edad y máscara de validez. Si solo se dispone de información radial, proporcionar predicciones radiales y su confianza. Limitar y normalizar valores; representar explícitamente las pistas ausentes. Conservar las observaciones actuales como referencia.
4. **Adaptar y volver a entrenar la política.** Ampliar y versionar el contrato de observaciones, el historial y el manifiesto de exportación. El checkpoint actual no acepta canales adicionales directamente. Preparar una inicialización compatible, conservando la correspondencia de las entradas antiguas de cada fotograma del historial y asignando pesos inicialmente nulos a las nuevas entradas en la primera capa. Comprobar que reproduce la política original antes de entrenar con las nuevas entradas.
5. **Medir la aportación real.** Comparar, con escenarios y presupuesto equivalentes, política actual, filtrado sin predicción y filtrado con predicción. Medir éxito, colisiones, error de predicción a cada horizonte, consistencia de incertidumbre, pérdida de pistas, latencia y coste de inferencia. Las posiciones reales del simulador se usarán para medir el error, nunca como entrada del filtro o de la red.

El seguimiento de obstáculos y la estimación de navegación propia son problemas relacionados pero distintos. Añadir Kalman para obstáculos no resuelve automáticamente el proveedor de objetivo relativo y velocidad que todavía procede del simulador.

### Revisar los sensores sin disparar el coste

La simulación actual tiene **26 sensores de distancia**. Cada uno aproxima un campo de visión de **27°** con nueve subrayos, y reduce los retornos a una única distancia mínima. Por tanto, ya existe una aproximación de cono: lo que falta es conservar información espacial dentro de ese campo de visión. La observación de este modelo proporciona una proximidad escalar por sensor e historial, sin identidad ni velocidad de objetos.

| Alternativa a estudiar | Qué aportaría | Qué hay que comprobar |
|---|---|---|
| Mantener ToF de una distancia por sensor | Base sencilla para filtrado y predicción radial | Ambigüedad angular, asociación de retornos y cobertura |
| Sustituir algunos por **ToF multizona** | Varias distancias distribuidas angularmente dentro del campo de visión | Resolución frente a frecuencia, alcance útil, luz ambiente, interferencias y carga del bus |
| Añadir un **radar FMCW** en una dirección prioritaria | Explorar mediciones de distancia, velocidad radial y, según el sistema, ángulo | Resolución angular, distancia mínima, blancos pequeños, multitrayecto, procesamiento, consumo y masa |

Como candidato multizona, el **VL53L5CX** ofrece rejillas de 4×4 u 8×8 zonas y campo de visión diagonal de 65°. Su resolución y frecuencia deben elegirse juntas; el alcance efectivo depende de las condiciones. Véase la [ficha técnica de ST](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf). Para estudiar radar, el [BGT60TR13C de Infineon](https://www.infineon.com/part/BGT60TR13C) es una referencia FMCW de 60 GHz; su idoneidad para este dron requiere evaluar el sistema completo y el procesamiento de detecciones.

**Propuesta inicial:** comparar primero el filtrado con el montaje actual y una configuración híbrida que sustituya solo algunos ToF por multizona en sectores prioritarios. Considerar radar si las pruebas muestran una ventaja suficiente. Un radar no es automáticamente más preciso en todas las dimensiones, y un cono más ancho con una única distancia no aporta la misma información que un sensor multizona.

Para mantener el objetivo de un dron barato, pequeño y ágil, la selección deberá comparar coste total de módulos, PCB, alimentación, buses, procesamiento y montaje, además de gramos, consumo y latencia. **Todavía no hay una lista de materiales ni un incremento de coste validado**; no se presupone que sustituir sensores mantenga el precio. Antes de entrenar, habrá que modelar las características y limitaciones del sensor elegido en Crazyflow y actualizar el contrato de entradas.
