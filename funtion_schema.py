from pydantic import BaseModel, model_validator
import json
from typing import Optional, List, Dict
import numpy as np
from llm_sdk import Small_LLM_Model as llm


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
        for _ in range(max_new_token):
            logits = self.model.get_logits_from_input_ids(inputs_ids)
            next_token_id = np.argmax(logits)
            next_token_str = self.model.decode([next_token_id])
            if next_token_str.strip() == "" or "\n" in next_token_str:
                break
            generated_name += next_token_str
            inputs_ids.append(int(next_token_id))
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
    def __init__(self, schema: FunctionSchema, model: llm, vocab: VocabManager):
        self.schemas = schema
        self.vocab = vocab
        self.model = model
        self.current_args = 0
        self.state = "Start"
        self.sequence_to_force = []
        self.ptr = 0

    def update_state(self, last_token_id: int) -> None:
        # Si estamos siguiendo una secuencia forzada, no cambiamos de estado 
        # hasta que el puntero llegue al final.
        if self.ptr < len(self.sequence_to_force):
            return 

        text = self.vocab.rvocabulary.get(last_token_id)
        
        # --- CASO 1: Estamos escribiendo el valor del PROMPT ---
        if self.state == "Prompt_value":
            if '"' in text:
                self.state = "Fn_Name_key"
                self.ptr = 0
                self.sequence_to_force = []

        # --- CASO 2: Estamos escribiendo el NOMBRE de la FUNCIÓN ---
        elif self.state == "Fn_Name_value":
            if '"' in text:
                self.state = "Args_Open"
                self.ptr = 0
                self.sequence_to_force = []

        # --- CASO 3: Estamos escribiendo el VALOR de un ARGUMENTO ---
        elif self.state == "Arg_Value":
            arg_name = self.schemas.args_names[self.current_args]
            arg_type = self.schemas.args_types[arg_name]

            should_transition = False

            if arg_type == "str":
                # Los strings terminan cuando el modelo genera la comilla de cierre
                if '"' in text:
                    should_transition = True
            else:
                # Int, Float o Bool NO llevan comillas en el JSON.
                # Terminan cuando el modelo genera algo que no es parte del valor
                # (como una coma para el siguiente arg o una llave para cerrar).
                if "," in text or "}" in text or " " in text:
                    should_transition = True

            if should_transition:
                # ¿Quedan más argumentos por rellenar?
                if self.current_args < len(self.schemas.args_names) - 1:
                    self.current_args += 1
                    self.state = "Arg_Name"
                else:
                    self.state = "End"
                
                self.ptr = 0
                self.sequence_to_force = []

    def get_next_mask(self, last_token_id: Optional[int] = None) -> Optional[List[int]]:
        """
        Docstring for get_next_mask

        :param self: Description
        :param last_token_id: Description
        :type last_token_id: Optional[int]
        :return: Description
        :rtype: Any
        """
        if last_token_id is not None:
            self.update_state(last_token_id)
        
        # SI HAY UNA SECUENCIA FORZADA EN CURSO:
        if self.ptr < len(self.sequence_to_force):
            next_token = self.sequence_to_force[self.ptr]
            self.ptr += 1
            return [next_token]  # Solo permitimos el token que dice el puntero

        # SI LA SECUENCIA TERMINÓ, CAMBIAMOS DE ESTADO Y CARGAMOS LA SIGUIENTE
        if self.state == "Start":
            self.state = "Prompt_value"
            # Pre-tokenizamos la apertura del JSON
            self.sequence_to_force = self.model.encode(
                '{"prompt": "')
            self.ptr = 1  # Ya devolvemos el primer token ahora
            return [self.sequence_to_force[0]]
        elif self.state == "Prompt_value":
            # Aquí no hay puntero, el modelo tiene libertad
            # (pero restringida a strings)
            return self.vocab.ids_str

        elif self.state == "Fn_Name_key":
            self.state = "Fn_Name_value"
            # Pre-tokenizamos el salto a la función
            self.sequence_to_force = self.model.encode(
                '", "function": "')
            self.ptr = 1
            return [self.sequence_to_force[0]]
        elif self.state == "Fn_Name_value":
            # Aquí el modelo escribe el nombre de la función (ej: fn_add_numbers)
            # Restringimos a los IDs que componen ese nombre específico
            return self.vocab.get_ids_from_string(self.schemas.fn_name)
        elif self.state == "Args_Open":
            # Forzamos la transición: ", "arguments": {
            self.sequence_to_force = self.model.encode('", "arguments": {')
            self.ptr = 1
            self.state = "Arg_Name" # Después de abrir, toca el primer nombre de argumento
            return [self.sequence_to_force[0]]

        elif self.state == "Arg_Name":
            arg_name = self.schemas.args_names[self.current_args]
            # Construimos la parte fija que toca escribir ahora
            prefix = ", " if self.current_args > 0 else ""
            text_to_force = f'{prefix}"{arg_name}": "'
            
            # Convertimos a IDs y preparamos el puntero
            self.sequence_to_force = self.model.encode(
                text_to_force)
            self.ptr = 1 # El primer token lo devolvemos ya
            self.state = "Arg_Value"  # Siguiente estado después de la secuencia
            return [self.sequence_to_force[0]]

        elif self.state == "Arg_Value":
            # 1. Identificamos el nombre y el tipo del argumento actual
            arg_name = self.schemas.args_names[self.current_args]
            arg_type = self.schemas.args_types[arg_name]

            # 2. Devolvemos los IDs permitidos según el tipo
            if arg_type == "int":
                return self.vocab.ids_ints
            elif arg_type == "float":
                # Asegúrate de tener ids_float en tu VocabManager
                return self.vocab.ids_float 
            elif arg_type == "bool":
                return self.vocab.ids_booleans
            else: # Por defecto tratamos como string
                return self.vocab.ids_str

        elif self.state == "End":
            self.sequence_to_force = self.model.encode('}}')
            self.ptr = 1
            return [self.sequence_to_force[0]]

    def generate_step(self, model, token_history):
        # 1. Obtenemos los logits brutos del modelo
        raw_logits = model.get_logits_from_input_ids(token_history)
        logits_np = np.array(raw_logits)

        # 2. Obtenemos la lista de tokens permitidos por la máquina de estados
        # Usamos el último token generado para actualizar el estado
        last_token = token_history[-1] if token_history else None
        allowed_ids = self.get_next_mask(last_token)

        # 3. Aplicamos la máscara (Logit Processing)
        # Ponemos todo a -infinito
        penalty_mask = np.full(len(logits_np), -float('inf'))
        
        # Solo los permitidos vuelven a tener su probabilidad original (0.0 de penalización)
        if allowed_ids:
            penalty_mask[allowed_ids] = 0.0
        
        masked_logits = logits_np + penalty_mask

        # 4. Elegimos el mejor token entre los permitidos
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