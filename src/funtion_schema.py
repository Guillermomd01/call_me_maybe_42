from pydantic import BaseModel, model_validator
import json
from typing import Optional, List, Dict
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model as llm


class FunctionSchema(BaseModel):
    """
    Class for reciving Json data and validate with pydantic
    """
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str

    @model_validator(mode='after')
    def validate_keys(self) -> 'FunctionSchema':
        """
        Validate if args_names are in args_types
        """
        if set(self.args_types.keys()) != set(self.args_names):
            raise ValueError("Names are not similar")
        return self


class VocabManager():
    """
    Docstring for VocabManager
    Creamos un manager para filtrar los tokens segun su tipo
    """
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
            self.ids_str.append(token_id) # Por defecto, casi todo es un string
            
            if token.lower() in ["true", "false"]:
                self.ids_booleans.append(token_id)
            
            # Capturar dígitos y el punto decimal para floats
            clean_token = token.strip().replace(' ', '')
            if clean_token.isdigit() or (clean_token.startswith('-') and clean_token[1:].isdigit()):
                self.ids_ints.append(token_id)
                self.ids_float.append(token_id)
            elif clean_token == "." or ("." in clean_token and any(c.isdigit() for c in clean_token)):
                self.ids_float.append(token_id)
            
            if token in self.signals:
                self.ids_structs.append(token_id)

    def get_ids_by_prefix(self, prefix: str) -> List[int]:
        """
        Docstring for get_ids_by_prefix

        :param self: Description
        :param prefix: prefijo que vamos a buscar dentro del vocabulario
        :type prefix: str
        :return: Lista de ids que coincidan con el prefijo
        :rtype: list[int]
        """
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
                self.functions_map = {
                    f["fn_name"]: FunctionSchema(**f) for f in data}
        except FileNotFoundError:
            print("File don't exist")

    def get_function_name(self, user_query: str) -> FunctionSchema | None:
        prompt = f"""You are a expert function picker
        Available Functions: {self.list_functions}.
        User query: {user_query}
        Return only the function name"""
        inputs_ids = self.model.encode(prompt)
        generated_name = ""
        max_new_token = 20
        for _ in range(20):
            logits = self.model.get_logits_from_input_ids(inputs_ids)
            next_token_id = int(np.argmax(logits))
            inputs_ids.append(next_token_id)
            next_token_str = self.model.decode([next_token_id])
            if next_token_str.strip() == "" or "\n" in next_token_str:
                break
            generated_name += next_token_str
        name = generated_name.strip().replace('"','').replace("'","")
        for fn_name in self.list_functions:
            if fn_name.lower() in name.lower() or name.lower() in fn_name.lower():
                return self.functions_map[fn_name]
        return None


class JsonGenerator():
    """
    Docstring for JsonGenerator
    vamos a hacer un clase que va a generar el jason final
    harcodeando la parte de la clave que siempre es igual
    y el valor recorriendo segun las listas que tenemos de vocab
    """
    def __init__(self, schema: FunctionSchema, model: llm, vocab: VocabManager, user_query: str):
        self.schemas = schema
        self.vocab = vocab
        self.model = model
        self.user_query = user_query 
        self.current_args = 0
        self.state = "Start"
        self.sequence_to_force = []
        self.ptr = 0

    def update_state(self, last_token_id: int) -> None:
        if self.ptr < len(self.sequence_to_force):
            return 

        text = self.vocab.rvocabulary.get(last_token_id, "")

        if self.state == "Arg_Value":
            arg_name = self.schemas.args_names[self.current_args]
            # Buscamos el tipo, por defecto string para evitar errores si no existe
            arg_type = self.schemas.args_types.get(arg_name, "str")

            should_transition = False

            if arg_type in ["str", "string"]:
                # Los strings terminan cuando el modelo genera la comilla de cierre
                if '"' in text:
                    should_transition = True
            else:
                # Int, Float o Bool terminan con coma, llave o espacio
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

    def get_next_mask(self, last_token_id: Optional[int] = None) -> Optional[List[int]]:
        if last_token_id is not None:
            self.update_state(last_token_id)
        
        if self.ptr < len(self.sequence_to_force):
            next_token = self.sequence_to_force[self.ptr]
            self.ptr += 1
            return [next_token]

        if self.state == "Start":
            self.state = "Arg_Name"
            
            # Escapamos las comillas del prompt original para no corromper el JSON
            safe_prompt = self.user_query.replace('"', '\\"')
            
            # Forzamos TODO el inicio exacto que pide el subject (prompt, fn_name y args)
            text_to_force = f'{{"prompt": "{safe_prompt}", "fn_name": "{self.schemas.fn_name}", "args": {{'
            
            self.sequence_to_force = self.model.encode(text_to_force)
            self.ptr = 1 
            return [self.sequence_to_force[0]]

        elif self.state == "Arg_Name":
            arg_name = self.schemas.args_names[self.current_args]
            prefix = ", " if self.current_args > 0 else ""
            text_to_force = f'{prefix}"{arg_name}": '
            
            if self.schemas.args_types.get(arg_name) in ["str", "string"]:
                text_to_force += '"'
                
            self.sequence_to_force = self.model.encode(text_to_force)
            self.ptr = 1 
            self.state = "Arg_Value"
            return [self.sequence_to_force[0]]

        elif self.state == "Arg_Value":
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

        elif self.state == "End":
            self.sequence_to_force = self.model.encode('}}')
            self.ptr = 1
            return [self.sequence_to_force[0]]
            
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

    def generate_full_json(self, model, picker, vocab, prompt_usuario):
        # 1. El Picker decide qué función usar
        schema = picker.get_function_name(prompt_usuario)
        if not schema:
            return ""
        # 2. Inicializamos el generador con el esquema elegido
        generator = JsonGenerator(schema, model, vocab)
        
        # 3. El historial de tokens empieza con el prompt del usuario
        # (Ojo: el subject suele pedir que el JSON sea una respuesta al prompt)
        token_history = model.encode(prompt_usuario)
        
        # 4. Bucle hasta que la máquina de estados llegue a "End"
        max_tokens = 512
        for _ in range(max_tokens):
            next_token = generator.generate_step(model, token_history)
            token_history.append(next_token)
            
            if generator.state == "End" and generator.ptr >= len(generator.sequence_to_force):
                break
                
        # 5. Decodificamos solo la parte del JSON generada
        # (Normalmente desde donde terminó el prompt del usuario)
        full_text = model.decode(token_history)
        return full_text