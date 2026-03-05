from pydantic import BaseModel, model_validator
import json
from typing import Optional, List, Dict
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
            if clean_token.isdigit() or (clean_token.startswith('-') and clean_token[1:].isdigit()):
                self.ids_ints.append(token_id)
                self.ids_float.append(token_id)
            elif clean_token == "." or ("." in clean_token and any(c.isdigit() for c in clean_token)):
                self.ids_float.append(token_id)
            
            if token in self.signals:
                self.ids_structs.append(token_id)

    def get_ids_by_prefix(self, prefix: str) -> List[int]:
        return [id for token, id in self.token_list if token.startswith(prefix)]

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
        prompt = f"""Task: Map the user query to the exact function name from the list.
Functions: {self.list_functions}

Query: "say hello"
Function: fn_greet

Query: "what is 2 + 2"
Function: fn_add_numbers

Query: "{user_query}"
Function: fn_"""

        inputs_ids = self.model.encode(prompt)
        generated_name = "fn_"
        
        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(inputs_ids)
            next_token_id = int(np.argmax(logits))
            inputs_ids.append(next_token_id)
            token_str = self.model.decode([next_token_id])
            
            generated_name += token_str
            # Permitimos espacios internos para que el modelo pueda expresarse sin cortarse de golpe
            if "\n" in token_str or "<" in token_str:
                break
                
        generated_clean = generated_name.strip().lower()
        
        for fn_name in self.list_functions:
            if fn_name.lower() in generated_clean or generated_clean in fn_name.lower():
                return self.functions_map[fn_name]
                
        # FALLBACK INTELIGENTE (Sin stopwords y añadiendo sinónimos al diccionario dinámico)
        clean_query = user_query.lower().replace('?', '').replace('.', '').replace("'", '').replace('"', '')
        stopwords = {"what", "is", "the", "of", "and", "a", "an", "to", "in", "with", "for", "on", "all"}
        query_words = set(clean_query.split()) - stopwords
        
        best_match = None
        max_overlap = 0
        
        for fn_name in self.list_functions:
            fn_words = set(fn_name.lower().replace('fn_', '').split('_'))
            if "add" in fn_words: fn_words.add("sum")
            if "multiply" in fn_words: fn_words.add("product")
            if "substitute" in fn_words: fn_words.add("replace")
            
            overlap = len(query_words.intersection(fn_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = fn_name
                
        if best_match and max_overlap > 0:
            return self.functions_map[best_match]

        return None


class JsonGenerator():
    def __init__(self, schema: FunctionSchema, model: llm, vocab: VocabManager, user_query: str):
        self.schemas = schema
        self.vocab = vocab
        self.model = model
        self.user_query = user_query 
        self.current_args = 0
        self.state = "Start"
        self.sequence_to_force = []
        self.ptr = 0
        self.is_llm_token = False  # <-- NUEVO: Control absoluto sobre quién escribe

    def get_next_mask(self, last_token_id: Optional[int] = None) -> Optional[List[int]]:
        # 1. Actualizar el estado SOLO si el token anterior fue generado por el LLM libremente
        if self.is_llm_token and last_token_id is not None:
            text = self.vocab.rvocabulary.get(last_token_id, "")
            arg_name = self.schemas.args_names[self.current_args]
            arg_type = self.schemas.args_types.get(arg_name, "str")

            should_transition = False
            if arg_type in ["str", "string"]:
                if '"' in text:
                    should_transition = True
            else:
                if "," in text or "}" in text or " " in text or "\n" in text:
                    should_transition = True

            if should_transition:
                if self.current_args < len(self.schemas.args_names) - 1:
                    self.current_args += 1
                    self.state = "Arg_Name"
                else:
                    self.state = "End"
                self.ptr = 0
                self.sequence_to_force = []
                self.is_llm_token = False

        # 2. Inicializar la secuencia que debemos forzar según el estado actual
        if self.ptr >= len(self.sequence_to_force):
            if self.state == "Start":
                safe_prompt = self.user_query.replace('"', '\\"')
                text_to_force = f'{{"prompt": "{safe_prompt}", "fn_name": "{self.schemas.fn_name}", "args": {{'
                self.sequence_to_force = self.model.encode(text_to_force)
                self.ptr = 0
            elif self.state == "Arg_Name":
                arg_name = self.schemas.args_names[self.current_args]
                prefix = ", " if self.current_args > 0 else ""
                text_to_force = f'{prefix}"{arg_name}": '
                if self.schemas.args_types.get(arg_name) in ["str", "string"]:
                    text_to_force += '"'
                self.sequence_to_force = self.model.encode(text_to_force)
                self.ptr = 0
            elif self.state == "End":
                self.sequence_to_force = self.model.encode('}}')
                self.ptr = 0

        # 3. Emitir los tokens que estamos forzando (sin dejar al LLM pensar)
        if self.ptr < len(self.sequence_to_force):
            next_token = self.sequence_to_force[self.ptr]
            self.ptr += 1
            self.is_llm_token = False
            
            # Si acabamos de emitir el ÚLTIMO token de nuestra secuencia, preparamos la transición
            if self.ptr >= len(self.sequence_to_force):
                if self.state == "Start":
                    # Salvaguarda por si hubiera alguna función sin argumentos
                    if len(self.schemas.args_names) > 0:
                        self.state = "Arg_Name"
                    else:
                        self.state = "End"
                elif self.state == "Arg_Name":
                    self.state = "Arg_Value"
                    
            return [next_token]

        # 4. Si no estamos forzando nada, dejamos que el LLM genere con su vocabulario permitido
        self.is_llm_token = True
        if self.state == "Arg_Value":
            arg_name = self.schemas.args_names[self.current_args]
            arg_type = self.schemas.args_types.get(arg_name, "str")

            if arg_type in ["int", "integer"]:
                return self.vocab.ids_ints + self.vocab.ids_structs
            elif arg_type in ["float", "number"]:
                return self.vocab.ids_float + self.vocab.ids_structs
            elif arg_type in ["bool", "boolean"]:
                return self.vocab.ids_booleans + self.vocab.ids_structs
            else: 
                return self.vocab.ids_str + self.vocab.ids_structs

        return None

    def generate_step(self, model, token_history):
        raw_logits = model.get_logits_from_input_ids(token_history)
        logits_np = np.array(raw_logits)
        last_token = token_history[-1] if token_history else None
        allowed_ids = self.get_next_mask(last_token)

        penalty_mask = np.full(len(logits_np), -float('inf'))
        if allowed_ids:
            penalty_mask[allowed_ids] = 0.0
        
        masked_logits = logits_np + penalty_mask
        next_token_id = int(np.argmax(masked_logits))
        return next_token_id