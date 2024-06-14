from typing import Any
from yaml_config_override import add_arguments # type: ignore


def args() -> dict[str, Any]:
    """
    Questa funzione ci permette di sovrascrivere il tipo di ritorno della funzione `add_arguments`
    """
    return add_arguments() # type: ignore
