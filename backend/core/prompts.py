"""
Intinya: Prompts("path/to/prompts.yaml") → dict yang isinya semua template prompt dari YAML file. 
Karena SingletonBaseClass, file yang sama hanya dibaca sekali.
"""

from pathlib import Path
import yaml
from core.utils import SingletonBaseClass

#  Singleton + dict: satu file YAML = satu instance, bisa diakses prompt_dict["key"]
class Prompts(SingletonBaseClass, dict[str, str]):
    def __init__(self, file_path: Path) -> None:
        super().__init__()
        with file_path.open(encoding="utf8") as file:
            prompt_yaml_dict = yaml.safe_load(file)
            # parse YAML file -> python dict

        if prompt_yaml_dict is None:
            error_message = f"Failed to load prompts from {file_path}"
            raise ValueError(error_message)

        for key, value in prompt_yaml_dict.items():
            self[key] = value
        # ^ setiap key di YAML jadi key di dict ini
        #   misal: prompt_dict["hypothesis_gen"]["system_prompt"] = "You are..."
