from pydantic import BaseModel, model_validator
import json
from typing import List, Dict
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model as llm


class FunctionSchema(BaseModel):
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str

    @model_validator(mode='after')
    def validate_keys(self) -> 'FunctionSchema':
        if set(self.args_types.keys()) != set(self.args_names):
            raise ValueError("Names are not similar")
        return self


class VocabManager():
    def __init__(self, path_vocabulary: str):
        with open(path_vocabulary) as file:
            self.vocabulary = json.load(file)
        self.token_list = list(self.vocabulary.items())
        self.rvocabulary = {v: k for k, v in self.vocabulary.items()}
        self.ids_ints = []
        self.ids_float = []
        self.ids_booleans = []
        self.ids_structs = []
        self.ids_str = []
        self.signals = [":", ",", '"', ' "', "{", "}"]

        for token, token_id in self.vocabulary.items():
            if "\n" not in token and "Ċ" not in token and "\\n" not in token:
                self.ids_str.append(token_id)

            if token.lower() in ["true", "false"]:
                self.ids_booleans.append(token_id)

            clean_token = token.strip().replace(' ', '')
            if clean_token.isdigit() or (
                    clean_token.startswith('-') and clean_token[1:].isdigit()):
                self.ids_ints.append(token_id)
                self.ids_float.append(token_id)
            elif clean_token == "." or (
                    "." in clean_token and any(
                        c.isdigit() for c in clean_token)):
                self.ids_float.append(token_id)

            if token in self.signals:
                self.ids_structs.append(token_id)

    def get_ids_by_prefix(self, prefix: str) -> List[int]:
        return [
            id for token, id in self.token_list if token.startswith(prefix)]

    def get_ids_from_string(self, target_text: str) -> List[int]:
        target_id = self.vocabulary.get(target_text)
        if target_id is not None:
            return [target_id]
        return [id for token, id in self.token_list if token == target_text]


class FunctionPicker():
    def __init__(self, json_path: str, model: llm):
        self.list_functions = []
        self.model = model
        try:
            with open(json_path, "r") as file:
                data = json.load(file)
                self.list_functions = [f["fn_name"] for f in data]
                self.functions_map = {f["fn_name"]: FunctionSchema(**f) for f in data}
        except FileNotFoundError:
            print("File don't exist")

    def get_function_name(self, user_query: str) -> FunctionSchema | None:
        prompt = (
            f"Task: Map the query to the function name.\n"
            f"Functions: {self.list_functions}\n"
            "Query: 'say hello' -> Function: fn_greet\n"
            "Query: 'sum 2 and 2' -> Function: fn_add_numbers\n"
            "Query: 'replace digits' -> Function: fn_substitute_string_with_regex\n"
            f"Query: '{user_query}' -> Function: fn_"
        )

        inputs_ids = self.model.encode(prompt)[0].tolist()
        generated_name = "fn_"
        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(inputs_ids)
            next_token_id = int(np.argmax(logits))
            inputs_ids.append(next_token_id)
            token_str = self.model.decode([next_token_id])
            generated_name += token_str
            if any(c in token_str for c in ["\n", "<", " "]): break
                
        generated_clean = generated_name.strip().lower()
        for fn_name in self.list_functions:
            if fn_name.lower() in generated_clean or generated_clean in fn_name.lower():
                return self.functions_map[fn_name]
                
        # FALLBACK INTELIGENTE
        clean_query = user_query.lower()
        stopwords = {"what", "is", "the", "of", "and", "a", "an", "to", "in", "for", "all"}
        query_words = set(clean_query.split()) - stopwords
        best_match, max_overlap = None, -1
        
        for fn_name in self.list_functions:
            fn_words = set(fn_name.lower().replace('fn_', '').split('_'))
            if "substitute" in fn_words: fn_words.update(["replace", "regex", "pattern"])
            overlap = len(query_words.intersection(fn_words))
            
            # Penalización fuerte para matemáticas si hay palabras de texto
            if any(w in query_words for w in ["replace", "regex", "string", "text"]):
                if "numbers" in fn_words or "root" in fn_words: overlap -= 5
            
            if overlap > max_overlap:
                max_overlap, best_match = overlap, fn_name
        return self.functions_map[best_match] if best_match else None

class JsonGenerator():
    def __init__(self, schema: FunctionSchema, model: llm, vocab: VocabManager, user_query: str):
        self.schema, self.model, self.vocab, self.user_query = schema, model, vocab, user_query

    def extract_arguments(self) -> dict:
        safe_query = self.user_query.replace('"', '\\"')
        # FIJAMOS EL CONTEXTO CON PARENTESIS Y EJEMPLOS (FEW-SHOT)
        context = (
            "Task: Extract parameters from query to JSON. Parameters must be raw strings from query.\n\n"
            "Query: 'Reverse the word apple'\n"
            "JSON: {\"prompt\": \"Reverse the word apple\", \"name\": \"fn_reverse_string\", \"parameters\": {\"s\": \"apple\"}}\n\n"
            "Query: 'Replace cat with dog in the cat sat'\n"
            "JSON: {\"prompt\": \"Replace cat with dog in the cat sat\", \"name\": \"fn_substitute_string_with_regex\", \"parameters\": {\"source_string\": \"the cat sat\", \"regex\": \"cat\", \"replacement\": \"dog\"}}\n\n"
            f"Query: '{safe_query}'\n"
            f"JSON: {{\"prompt\": \"{safe_query}\", \"name\": \"{self.schema.fn_name}\", \"parameters\": {{"
        )

        tokens = self.model.encode(context)[0].tolist()
        result = {}
        max_id = max(self.vocab.rvocabulary.keys())
        vocab_list = [""] * (max_id + 1)
        for t_id, t_str in self.vocab.rvocabulary.items(): vocab_list[t_id] = t_str

        for i, arg_name in enumerate(self.schema.args_names):
            arg_type = self.schema.args_types.get(arg_name, "str")
            prefix = f'"{arg_name}": ' if i == 0 else f', "{arg_name}": '
            if arg_type in ["string", "str"]: prefix += '"'
            tokens.extend(self.model.encode(prefix)[0].tolist())

            value_str, tokens_generated = "", 0
            while tokens_generated < 30:
                tokens_generated += 1
                logits = np.array(self.model.get_logits_from_input_ids(tokens))
                for t_id in range(len(vocab_list)):
                    txt = vocab_list[t_id].replace('Ġ', ' ')
                    valid = True
                    if arg_type in ["number", "float", "integer", "int"]:
                        valid = all(c in "-0123456789. " for c in txt) or txt.strip() in [",", "}"]
                    if not valid: logits[t_id] = float("-inf")
                
                new_token = int(np.argmax(logits))
                tokens.append(new_token)
                new_text = vocab_list[new_token].replace('Ġ', ' ')
                if arg_type in ["string", "str"]:
                    if '"' in new_text:
                        value_str += new_text.split('"')[0]
                        break
                elif "," in new_text or "}" in new_text: break
                value_str += new_text

            v = value_str.strip()
            if arg_type in ["number", "float"]: result[arg_name] = float(v) if v else 0.0
            elif arg_type in ["integer", "int"]: result[arg_name] = int(v) if v else 0
            else: result[arg_name] = v
        return result