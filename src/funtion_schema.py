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
                self.list_functions = [
                    f["fn_name"] for f in data]
                self.functions_map = {
                    f["fn_name"]: FunctionSchema(**f) for f in data}
        except FileNotFoundError:
            print("File don't exist")

    def get_function_name(self,
                          user_query: str) -> FunctionSchema | None:
        prompt = f"""Task: Map the user query to the exact
        function name from the list.
        Functions: {self.list_functions}
        Query: "say hello"
        Function: fn_greet

        Query: "what is 2 + 2"
        Function: fn_add_numbers

        Query: "replace spaces with dashes in text"
        Function: fn_substitute_string_with_regex

        Query: "{user_query}"
        Function: fn_"""

        inputs_ids = self.model.encode(prompt)[0].tolist()
        generated_name = "fn_"

        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(inputs_ids)
            next_token_id = int(np.argmax(logits))
            inputs_ids.append(next_token_id)
            token_str = self.model.decode([next_token_id])

            generated_name += token_str
            if "\n" in token_str or "<" in token_str or " " in token_str:
                break

        generated_clean = generated_name.strip().lower()
        for fn_name in self.list_functions:
            aux = generated_clean in fn_name.lower()
            if fn_name.lower() in generated_clean or aux:
                return self.functions_map[fn_name]

        clean_query = user_query.lower().replace('?', '').replace('.', '')
        f_clean_query = clean_query.replace("'", '').replace('"', '')
        stopwords = {
            "what", "is", "the", "of", "and", "a",
            "an", "to", "in", "with", "for", "on", "all"}
        query_words = set(f_clean_query.split()) - stopwords

        best_match = None
        max_overlap = -1

        for fn_name in self.list_functions:
            fn_words = set(fn_name.lower().replace('fn_', '').split('_'))
            if "add" in fn_words:
                fn_words.add("sum")
            if "multiply" in fn_words:
                fn_words.add("product")

            if "substitute" in fn_words:
                fn_words.update(["replace", "regex", "pattern", "vowels"])

            overlap = len(query_words.intersection(fn_words))

            if "replace" in query_words and fn_name == "fn_add_numbers":
                overlap -= 2

            if overlap > max_overlap:
                max_overlap = overlap
                best_match = fn_name

        if best_match and max_overlap >= 0:
            return self.functions_map[best_match]

        return None


class JsonGenerator():
    def __init__(self, schema: FunctionSchema, model: llm,
                 vocab: VocabManager, user_query: str):
        self.schema = schema
        self.model = model
        self.vocab = vocab
        self.user_query = user_query

    def extract_arguments(self) -> dict[
            str, int | float | bool | str | None]:
        """
        Extrae los argumentos forzando la generación token a
        token usando Logit Masking.
        Devuelve directamente un diccionario de Python válido.
        """
        # 1. Forzamos el contexto inicial
        safe_query = self.user_query.replace('"', '\\"')
        context = f'{{"prompt": "{safe_query}",'
        f'"fn_name": "{self.schema.fn_name}", "args": {{'

        # Extraemos la lista plana de tokens del tensor
        tokens = self.model.encode(context)[0].tolist()
        result: dict[str, int | float | bool | str | None] = {}

        # Pre-calculamos la lista del vocabulario para acceso rápido por índice
        max_id = max(self.vocab.rvocabulary.keys())
        vocab_list = [""] * (max_id + 1)
        for token_id, token_str in self.vocab.rvocabulary.items():
            vocab_list[token_id] = token_str

        # 2. Iteramos por cada parámetro que requiera la función
        for i, arg_name in enumerate(self.schema.args_names):
            arg_type = self.schema.args_types.get(arg_name, "str")

            # Forzamos la escritura de la clave del JSON
            prefix = f'"{arg_name}": ' if i == 0 else f', "{arg_name}": '
            if arg_type in ["string", "str"]:
                prefix += '"'

            tokens.extend(self.model.encode(prefix)[0].tolist())

            value_str = ""
            max_tokens_per_arg = 20
            tokens_generated = 0

            # 3. Bucle de generación bloqueando logits según el tipo
            while tokens_generated < max_tokens_per_arg:
                tokens_generated += 1
                logits = np.array(self.model.get_logits_from_input_ids(tokens))

                # Enmascaramiento de logits (Logit Masking)
                for token_id in range(len(vocab_list)):
                    token_text = vocab_list[token_id].replace('Ġ', ' ')
                    is_valid = True

                    if arg_type in ["number", "float"]:
                        is_valid = all(
                            c in "-0123456789. " for c in token_text)
                        if token_text.strip() in [",", "}"]:
                            is_valid = True

                    elif arg_type in ["integer", "int"]:
                        is_valid = all(c in "-0123456789 " for c in token_text)
                        if token_text.strip() in [",", "}"]:
                            is_valid = True

                    elif arg_type in ["boolean", "bool"]:
                        # Permitimos las letras de true/false o los cierres
                        is_valid = token_text.strip().lower() in [
                            "true", "false", ",", "}"
                            ] or token_text.strip() == ""
                        if any(w.startswith(
                                token_text.strip().lower()
                                ) for w in ["true", "false"]):
                            is_valid = True

                    # Si el token no es válido para este tipo de dato,
                    # probabilidad 0 (log(0) = -inf)
                    if not is_valid:
                        logits[token_id] = float("-inf")

                # Seguridad por si el vocabulario del
                # modelo es mayor que nuestra lista
                for j in range(len(vocab_list), len(logits)):
                    logits[j] = float("-inf")

                # Elegimos el mejor token permitido y lo añadimos
                new_token = int(np.argmax(logits))
                tokens.append(new_token)
                new_text = vocab_list[new_token].replace('Ġ', ' ')

                # 4. Condiciones de parada
                if arg_type in ["string", "str"]:
                    if '"' in new_text:
                        value_str += new_text.split('"')[0]
                        break
                else:
                    if "," in new_text or "}" in new_text:
                        break

                value_str += new_text

            # 5. Conversión segura (Casting)
            clean_val = value_str.strip()
            try:
                if arg_type in ["number", "float"]:
                    result[arg_name] = float(clean_val) if clean_val else 0.0
                elif arg_type in ["integer", "int"]:
                    result[arg_name] = int(clean_val) if clean_val else 0
                elif arg_type in ["boolean", "bool"]:
                    result[arg_name] = clean_val.lower() == "true"
                else:
                    result[arg_name] = clean_val
            except ValueError:
                result[arg_name] = None

        return result
