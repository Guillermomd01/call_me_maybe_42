*Este proyecto ha sido creado como parte del currículo de 42 por gumunoz.*

## Descripción
**Call Me Maybe** es un proyecto de Inteligencia Artificial enfocado en la implementación de **Function Calling** (llamada a funciones) utilizando modelos de lenguaje de pequeño tamaño (SLMs). El objetivo principal es transformar consultas de lenguaje natural en estructuras JSON precisas y validadas que puedan ser interpretadas por sistemas externos.

A diferencia de los modelos grandes (LLMs) que confían en su capacidad de razonamiento, este proyecto utiliza técnicas de **decodificación restringida** para garantizar que un modelo de apenas 0.6B parámetros genere siempre una salida sintácticamente perfecta y semánticamente coherente con las funciones definidas.

## Instrucciones

### Requisitos previos
* Python 3.10+
* Administrador de dependencias `uv` (recomendado)

### Instalación
1. Clona el repositorio.
2. Instala las dependencias y el SDK local:
   ```bash
   make
   ```

### Ejecución
Para procesar el archivo de pruebas por defecto y generar los resultados:
```bash
make run
```

## Recursos
* **Documentación:** Hugging Face Transformers, PyTorch, Pydantic.
* **Artículos:** *Constrained Decoding for NLP tasks*.
* **Uso de IA:** Se ha utilizado **Gemini** como asistente de IA para las siguientes tareas:
    * Depuración de errores de tipado estático y cumplimiento de normas de estilo.
    * Optimización del *prompt engineering* para mejorar la precisión en modelos de baja escala.
    *Optimización del README

## Explicación del Algoritmo: Decodificación Restringida
El núcleo del proyecto es el algoritmo de **Logit Masking** implementado en la clase `JsonGenerator`.

En lugar de permitir que el modelo elija cualquier palabra de su vocabulario, intervenimos en el proceso de generación token a token:
1. **Identificación de tipo:** Consultamos el esquema de la función (`FunctionSchema`) para saber qué tipo de dato se espera (int, float, string, bool).
2. **Máscara de Probabilidades:** Antes de realizar el `argmax`, recorremos el vocabulario del modelo. Si un token no cumple con el formato esperado (por ejemplo, una letra cuando se espera un número), su probabilidad (logit) se ajusta a $-\infty$.
3. **Selección forzada:** El modelo se ve obligado a elegir el token más probable de entre el subconjunto de tokens válidos, garantizando la integridad del JSON resultante.

## Decisiones de Diseño
* **Pydantic para Esquemas:** Se utiliza para validar que las definiciones de las funciones sean correctas antes de iniciar la ejecución.
* **Enrutamiento en dos pasos:** Primero, un `FunctionPicker` identifica la función objetivo mediante *Few-shot prompting*. Segundo, el `JsonGenerator` extrae los parámetros.
* **VocabManager:** Centraliza la clasificación de los miles de tokens del modelo para que las operaciones de enmascaramiento sean eficientes en tiempo de ejecución.

## Análisis de Rendimiento
* **Precisión:** Elevada gracias al uso de ejemplos en el prompt (*Few-shot*), que guían al modelo de 0.6B para que no "alucine" valores por defecto.
* **Velocidad:** El modelo es ligero, permitiendo inferencias casi instantáneas en CPU.
* **Fiabilidad:** La estructura del JSON es 100% robusta; el sistema es incapaz de generar un JSON mal formado gracias a que el código inyecta manualmente la sintaxis de control (`{`, `:`, `,`).

## Retos Encontrados
* **Fragilidad del JSON:** Inicialmente, el modelo generaba texto libre que requería limpiezas complejas con Regex. Se resolvió pasando a una generación guiada por el código.
* **Extracción numérica:** Los modelos pequeños tienden a repetir valores comunes (como 1.0 o 2.0). Se solucionó implementando un contexto con ejemplos específicos de extracción.
* **Tipado Estricto:** Cumplir con `mypy --disallow-untyped-defs` requirió una gestión minuciosa de los tipos de retorno en las funciones asíncronas y generadores.

## Estrategia de Pruebas
La validación se ha realizado mediante:
1. **Tests Automáticos:** Procesamiento de `function_calling_tests.json` y comparación manual con los resultados esperados.
2. **Análisis Estático:**
   * `flake8`: Para garantizar el cumplimiento de la norma PEP8.
   * `mypy`: Para verificar la seguridad de tipos en todo el proyecto.

## Ejemplos de Uso
**Consulta:** *"What is the sum of 10 and 25?"*
**Salida Generada:**
```json
{
    "prompt": "What is the sum of 10 and 25?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 10.0,
        "b": 25.0
    }
}
```