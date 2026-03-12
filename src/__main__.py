import json
import os
from typing import List, Any, Dict
from src.funtion_schema import VocabManager, FunctionPicker, JsonGenerator
from llm_sdk.llm_sdk import Small_LLM_Model as llm


def main() -> None:
    INPUT_FILE = "data/input/function_calling_tests.json"
    OUTPUT_FILE = "data/output/function_calling_results.json"
    FUNCTIONS_DEF = "data/input/function_definitions.json"

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading model...")
    model = llm()
    v_path = model.get_path_to_vocab_file()
    vocab = VocabManager(v_path)
    picker = FunctionPicker(FUNCTIONS_DEF, model)

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            tests = json.load(f)
    except Exception as e:
        print(f"Error reading entry: {e}")
        return

    results: List[Dict[str, Any]] = []
    for i, test in enumerate(tests):
        user_query = test["prompt"]
        print(f"[{i+1}/{len(tests)}] Processing: '{user_query}'")
        try:
            schema = picker.get_function_name(user_query)
            if not schema:
                continue

            gen = JsonGenerator(schema, model, vocab, user_query)
            args = gen.extract_arguments()

            results.append({
                "prompt": user_query,
                "name": schema.fn_name,
                "parameters": args
            })
        except Exception as e:
            print(f" Error in test {i}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=4)
    print(f"\nGeneration finished. Results in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
