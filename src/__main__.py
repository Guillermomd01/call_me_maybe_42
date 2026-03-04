import json
import os
import argparse
import numpy as np
from .funtion_schema import VocabManager, FunctionPicker, JsonGenerator
from llm_sdk.llm_sdk import Small_LLM_Model as llm

def main():
    # 1. Configuración de Argumentos (Requisito PDF) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    FUNCTIONS_DEF = "data/input/function_definitions.json" # Ajustado al PDF 
    
    # Crear carpeta de salida si no existe
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Inicialización
    print("Cargando modelo...")
    model = llm()
    vocab_path = model.get_path_to_vocab_file() # Nombre corregido [cite: 11]
    vocab = VocabManager(vocab_path)
    picker = FunctionPicker(FUNCTIONS_DEF, model)

    # 3. Carga de tests
    try:
        with open(args.input, "r") as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Error al leer entrada: {e}")
        return

    results = [] # Lista para el JSON final 

    # 4. Bucle de ejecución
    for i, test in enumerate(tests):
        user_query = test["prompt"]
        print(f"[{i+1}/{len(tests)}] Procesando: '{user_query}'")

        try:
            schema = picker.get_function_name(user_query)
            if not schema:
                continue

            generator = JsonGenerator(schema, model, vocab, user_query)
            token_history = model.encode(user_query)

            # Generación restringida
            for _ in range(512):
                next_token = generator.generate_step(model, token_history)
                token_history.append(next_token)

                if generator.state == "End" and generator.ptr >= len(generator.sequence_to_force):
                    break

            # Decodificación y limpieza
            full_output = model.decode(token_history)
            json_start = full_output.find('{')
            if json_start != -1:
                clean_json_str = full_output[json_start:]
                
                # Cortar todo lo que esté después de la última llave de cierre
                json_end = clean_json_str.rfind('}')
                if json_end != -1:
                    clean_json_str = clean_json_str[:json_end+1]
                
                # --- NUEVO: Limpiar impurezas generadas por el LLM ---
                import re
                # 1. Eliminar comas antes de una llave de cierre (ej: {"name": "shrek",} -> {"name": "shrek"})
                clean_json_str = re.sub(r',\s*}', '}', clean_json_str)
                
                # 2. Asegurar que termina exactamente con dos llaves de cierre }}
                # (una cierra el diccionario de args y la otra el JSON principal)
                clean_json_str = re.sub(r'}+\s*$', '}}', clean_json_str)
                
                # Formatear según el PDF: prompt, fn_name, args 
                generated_data = json.loads(clean_json_str)
                
                results.append({
                    "prompt": user_query,
                    "fn_name": schema.fn_name,
                    "args": generated_data.get("args", {}) 
                })

        except Exception as e:
            print(f" Error en test {i}: {e}")

    # 5. Guardar resultado único (Requisito PDF) 
    with open(args.output, "w") as out_f:
        json.dump(results, out_f, indent=4)
    print(f"\nGeneración finalizada. Resultados en: {args.output}")

if __name__ == "__main__":
    main()