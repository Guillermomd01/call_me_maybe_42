import json
import os
import numpy as np
from funtion_schema import VocabManager, FunctionPicker, JsonGenerator
from llm_sdk import Small_LLM_Model as llm


def main():
    # 1. Rutas de archivos y carpetas
    # Asegúrate de que estas rutas coinciden con tu estructura
    INPUT_TESTS = "data/input/function_calling_tests.json"
    FUNCTIONS_DEF = "function_definition/functions_definition.json"
    OUTPUT_DIR = "data/output"

    # Crear carpeta de salida si no existe (Requisito de robustez)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directorio creado: {OUTPUT_DIR}")

    # 2. Inicialización de los componentes
    print("Cargando modelo y vocabulario...")
    model = llm()
    
    # Suponiendo que el modelo tiene un método para darte la ruta del vocabulario
    # Si no, pon la ruta directa: "path/to/vocab.json"
    vocab_path = model.get_path_to_vocabulary_json()
    vocab = VocabManager(vocab_path)
    
    # El Picker decidirá qué función usar para cada prompt
    picker = FunctionPicker(FUNCTIONS_DEF, model)

    # 3. Carga de los casos de prueba
    try:
        with open(INPUT_TESTS, "r") as f:
            tests = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de tests en {INPUT_TESTS}")
        return

    print(f"Iniciando generación para {len(tests)} tests...\n")

    # 4. Bucle principal de ejecución
    for i, test in enumerate(tests):
        user_query = test["prompt"]
        print(f"[{i+1}/{len(tests)}] Procesando: '{user_query}'")

        try:
            # A. El Picker selecciona el esquema (FunctionSchema)
            schema = picker.get_function_name(user_query)
            
            if schema is None:
                print(f"   ! No se encontró una función para el prompt {i}")
                continue

            # B. Inicializamos el generador con el esquema encontrado
            generator = JsonGenerator(schema, model, vocab)

            # C. Historial de tokens inicial (el prompt del usuario)
            token_history = model.encode(user_query)

            # D. Generación token a token (Restricted Decoding)
            # El bucle se detiene cuando el estado es 'End' y no quedan secuencias fijas
            max_tokens = 512
            for _ in range(max_tokens):
                # Usamos el método que ya tienes en tu clase
                next_token = generator.generate_step(model, token_history)
                token_history.append(next_token)

                # Condición de parada: Máquina de estados en el final
                if generator.state == "End" and generator.ptr >= len(generator.sequence_to_force):
                    break

            # E. Decodificación y guardado
            full_output = model.decode(token_history)
            
            # Tip de 42: Extraer solo el JSON si el modelo repitió el prompt
            # Normalmente buscamos el primer '{'
            json_start = full_output.find('{')
            clean_json_str = full_output[json_start:]

            # Validamos que el JSON sea correcto antes de guardarlo
            final_data = json.loads(clean_json_str)
            
            output_file = os.path.join(OUTPUT_DIR, f"test_{i}.json")
            with open(output_file, "w") as out_f:
                json.dump(final_data, out_f, indent=4)
            
            print(f"   ✓ Éxito: Guardado en {output_file}")

        except Exception as e:
            print(f"   ✗ Error en test {i}: {e}")

    print("\nGeneración finalizada con éxito.")

if __name__ == "__main__":
    main()