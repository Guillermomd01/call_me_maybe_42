import json
from typing import List, Dict, Optional, Any
import numpy as np
from pydantic import BaseModel, model_validator
from llm_sdk.llm_sdk import Small_LLM_Model as llm


class FunctionSchema(BaseModel):
    """Defines the expected structure for function
    definitions, using Pydantic to ensure the integrity
    of data loaded from the JSON file"""
    fn_name: str
    args_names: List[str]
    args_types: Dict[str, str]
    return_type: str

    @model_validator(mode='after')
    def validate_keys(self) -> 'FunctionSchema':
        """Internal validator that runs after object
        creation to ensure every argument listed in args_names
        has a corresponding type defined in args_types"""
        if set(self.args_types.keys()) != set(self.args_names):
            raise ValueError("Names are not similar")
        return self


class VocabManager:
    """Manages the model's vocabulary and pre-classifies
    tokens into categories (integers, floats, booleans, etc.)
    to facilitate masking during generation."""
    def __init__(self, path_vocabulary: str) -> None:
        """Loads the vocabulary file, reverses the mapping
        for fast lookups, and iterates through tokens to
        identify and group them by their nature (numeric,
        boolean, or plain text)"""
        with open(path_vocabulary, encoding="utf-8") as file:
            self.vocabulary: Dict[str, int] = json.load(file)
        self.rvocabulary = {v: k for k, v in self.vocabulary.items()}
        self.ids_ints: List[int] = []
        self.ids_float: List[int] = []
        self.ids_booleans: List[int] = []
        self.ids_str: List[int] = []
        self.signals = [":", ",", '"', ' "', "{", "}"]

        for token, token_id in self.vocabulary.items():
            if all(x not in token for x in ["\n", "Ċ", "\\n"]):
                self.ids_str.append(token_id)
            if token.lower() in ["true", "false"]:
                self.ids_booleans.append(token_id)
            clean = token.strip().replace(' ', '')
            if clean.isdigit() or (clean.startswith('-') and
                                   clean[1:].isdigit()):
                self.ids_ints.append(token_id)
                self.ids_float.append(token_id)
            elif clean == "." or ("." in clean and
                                  any(c.isdigit() for c in clean)):
                self.ids_float.append(token_id)


class FunctionPicker:
    """Responsible for the routing phase, determining
    which function from the catalog best fits the user's
    request."""
    def __init__(self, json_path: str, model: llm) -> None:
        """Reads the function definitions file and
        builds a schema map for later consultation."""
        self.list_functions: List[str] = []
        self.functions_map: Dict[str, FunctionSchema] = {}
        self.model = model
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                self.list_functions = [f["fn_name"] for f in data]
                self.functions_map = {
                    f["fn_name"]: FunctionSchema(**f) for f in data
                }
        except FileNotFoundError:
            print("File don't exist")

    def get_function_name(self, user_query: str) -> Optional[FunctionSchema]:
        """Performs an initial inference with the model to predict the
        function name and applies a keyword-based fallback system to
        improve reliability in small models."""
        prompt = (
            f"Task: Map query to function name.\n"
            f"Functions: {self.list_functions}\n"
            "Query: 'sum 2 and 3' -> Function: fn_add_numbers\n"
            "Query: 'is 4 even' -> Function: fn_is_even\n"
            f"Query: '{user_query}' -> Function: fn_"
        )
        ids = self.model.encode(prompt)[0].tolist()
        gen_name = "fn_"
        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(ids)
            nt = int(np.argmax(logits))
            ids.append(nt)
            ts = self.model.decode([nt])
            gen_name += ts
            if any(c in ts for c in ["\n", "<", " "]):
                break

        res = gen_name.strip().lower()
        for name in self.list_functions:
            if name.lower() in res or res in name.lower():
                return self.functions_map[name]

        q_words = set(user_query.lower().split())
        best_match, max_overlap = None, -1.0
        for name in self.list_functions:
            f_w = set(name.lower().replace('fn_', '').split('_'))
            if "add" in f_w:
                f_w.update(["sum", "plus", "add"])
            if "substitute" in f_w:
                f_w.update(["replace", "regex", "vowels", "digits"])
            if "multiply" in f_w:
                f_w.update(["product", "times"])

            overlap = float(len(q_words.intersection(f_w)))
            if "sum" in q_words and name == "fn_add_numbers":
                overlap += 5.0
            if overlap > max_overlap:
                max_overlap, best_match = overlap, name
        return self.functions_map[best_match] if best_match else None


class JsonGenerator:
    """Core engine for constrained decoding that
    extracts parameters from the user query in a structured format."""
    def __init__(self, schema: FunctionSchema, model: llm,
                 vocab: VocabManager, query: str) -> None:
        """Sets up the generation session with the target
        function schema, the model to be used, and the original query."""
        self.schema, self.model, self.vocab, self.query = (
            schema, model, vocab, query)

    def extract_arguments(self) -> Dict[str, Any]:
        """Implements the token-by-token generation loop,
        injecting the JSON structure and applying Logit Masking
        to the model's logits to prohibit tokens that do not match
        the expected data type."""
        safe_q = self.query.replace('"', '\\"')
        context = (
            "Task: Extract parameters to JSON.\n\n"
            "Query: 'Sum 2 and 3'\n"
            "JSON: {\"prompt\": \"Sum 2 and 3\", \"name\": "
            "\"fn_add_numbers\", \"parameters\": {\"a\": 2.0, \"b\": 3.0}}\n\n"
            f"Query: '{safe_q}'\nJSON: {{\"prompt\": \"{safe_q}\", "
            f"\"name\": \"{self.schema.fn_name}\", \"parameters\": {{"
        )
        tokens = self.model.encode(context)[0].tolist()
        result: Dict[str, Any] = {}
        v_list = [""] * (max(self.vocab.rvocabulary.keys()) + 1)
        for t_id, t_str in self.vocab.rvocabulary.items():
            v_list[t_id] = t_str

        for i, arg_name in enumerate(self.schema.args_names):
            arg_type = self.schema.args_types.get(arg_name, "str")
            pref = f'"{arg_name}": ' if i == 0 else f', "{arg_name}": '
            if arg_type in ["string", "str"]:
                pref += '"'
            tokens.extend(self.model.encode(pref)[0].tolist())

            val_str, count = "", 0
            while count < 30:
                count += 1
                logits = np.array(self.model.get_logits_from_input_ids(tokens))
                for t_id in range(len(v_list)):
                    txt = v_list[t_id].replace('Ġ', ' ')
                    if arg_type in ["number", "float", "integer", "int"]:
                        if not (all(c in "-0123456789. " for c in txt) or
                                txt.strip() in [",", "}"]):
                            logits[t_id] = float("-inf")
                nt = int(np.argmax(logits))
                tokens.append(nt)
                new_txt = v_list[nt].replace('Ġ', ' ')
                if arg_type in ["string", "str"] and '"' in new_txt:
                    val_str += new_txt.split('"')[0]
                    break
                if arg_type not in ["string", "str"] and any(
                        c in new_txt for c in ",}"):
                    break
                val_str += new_txt

            v = val_str.strip()
            if arg_type in ["number", "float"]:
                result[arg_name] = float(v) if v else 0.0
            elif arg_type in ["integer", "int"]:
                result[arg_name] = int(v) if v else 0
            else:
                result[arg_name] = v
        return result
