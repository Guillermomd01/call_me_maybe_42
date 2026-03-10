import json
import os
import argparse
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

            # Usamos el nuevo generador basado en Logit Masking
            generator = JsonGenerator(schema, model, vocab, user_query)
            extracted_args = generator.extract_arguments()
            
            results.append({
                "prompt": user_query,
                "fn_name": schema.fn_name,
                "args": extracted_args 
            })

        except Exception as e:
            print(f" Error general en test {i}: {e}")

    with open(args.output, "w") as out_f:
        json.dump(results, out_f, indent=4)
    print(f"\nGeneración finalizada. Resultados en: {args.output}")

if __name__ == "__main__":
    main()