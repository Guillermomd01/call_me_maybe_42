import json
import os
import argparse
import numpy as np
import re
import ast
from .funtion_schema import VocabManager, FunctionPicker, JsonGenerator
from llm_sdk.llm_sdk import Small_LLM_Model as llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    args = parser.parse_args()

    FUNCTIONS_DEF = "data/input/function_definitions.json" 
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Cargando modelo...")
    model = llm()
    vocab_path = model.get_path_to_vocab_file() 
    vocab = VocabManager(vocab_path)
    picker = FunctionPicker(FUNCTIONS_DEF, model)

    try:
        with open(args.input, "r") as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Error al leer entrada: {e}")
        return

    results = []

    for i, test in enumerate(tests):
        user_query = test["prompt"]
        print(f"[{i+1}/{len(tests)}] Procesando: '{user_query}'")

        try:
            schema = picker.get_function_name(user_query)
            if not schema:
                print(f"  [!] No se pudo determinar la función. Saltando test...")
                continue

            generator = JsonGenerator(schema, model, vocab, user_query)
            prompt_context = f"Extract parameters to JSON.\nQuery: {user_query}\nJSON:\n"
            token_history = model.encode(prompt_context)

            for _ in range(512):
                next_token = generator.generate_step(model, token_history)
                token_history.append(next_token)

                if generator.state == "End" and generator.ptr >= len(generator.sequence_to_force):
                    break

            full_output = model.decode(token_history)
            json_start = full_output.find('{')
            
            if json_start != -1:
                clean_json_str = full_output[json_start:]
                
                json_end = clean_json_str.rfind('}')
                if json_end != -1:
                    clean_json_str = clean_json_str[:json_end+1]
                
                # --- MAGIA ANTIFALLOS PARA STRINGS SIN CERRAR ---
                # Si el LLM se olvidó de poner la comilla de cierre y hay un número impar de comillas
                if clean_json_str.count('"') % 2 != 0:
                    last_brace = clean_json_str.rfind('}')
                    if last_brace != -1:
                        # Inyectamos la comilla justo antes de que se cierre el diccionario
                        clean_json_str = clean_json_str[:last_brace] + '"' + clean_json_str[last_brace:]
                
                # Limpiamos dobles comas o comas pegadas a llaves
                clean_json_str = re.sub(r',\s*}', '}', clean_json_str)
                clean_json_str = re.sub(r',\s*,', ',', clean_json_str)
                
                generated_data = {}
                try:
                    generated_data = json.loads(clean_json_str)
                except Exception:
                    try:
                        eval_str = clean_json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                        generated_data = ast.literal_eval(eval_str)
                    except Exception as e:
                        print(f" Error irrecuperable al limpiar la respuesta: {e}")
                        continue
                
                results.append({
                    "prompt": user_query,
                    "fn_name": schema.fn_name,
                    "args": generated_data.get("args", {}) 
                })

        except Exception as e:
            print(f" Error en test {i}: {e}")

    with open(args.output, "w") as out_f:
        json.dump(results, out_f, indent=4)
    print(f"\nGeneración finalizada. Resultados en: {args.output}")

if __name__ == "__main__":
    main()