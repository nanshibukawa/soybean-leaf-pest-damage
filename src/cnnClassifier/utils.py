import traceback
import yaml



from box.exceptions import BoxValueError
from box import ConfigBox
from pathlib import Path
from pydantic import BaseModel

from cnnClassifier.logger import configure_logger

logger = configure_logger(__name__)

def read_yaml(base_path: Path) -> ConfigBox:
    """
    Returns:
        ConfigBox: Content of the YAML file wrapped in a ConfigBox. 
        If the file cannot be read, returns an empty ConfigBox.

    Raises:
    FineNotFoundError: If the file does not exist.
    BoxValueError: If the content cannot be converted to ConfigBox.
    """

    if not base_path.exists():
        logger.error(f"YAML file not found at {base_path}")
        raise FileNotFoundError
    
    try:
        with open(base_path, "r", encoding='utf-8') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.debug(f"YAML file read successfully from {base_path}.")

            if content is None:
                logger.warning(f"YAML file ay {base_path} is empty")
                return ConfigBox({})
            if isinstance(content, dict):
                return ConfigBox(content)
            elif isinstance(content,list):
                return ConfigBox({"items": content})
            else:
                logger.warning(f"Unexpected content type: {type(content)}")
                return ConfigBox({"data": content})

    except BoxValueError as e:
        logger.exception(f"BoxValueError reading {base_path}. Error: {str(e)}")
        return ConfigBox({})

    except Exception as e:
        logger.exception(f"Unexpected error reading {base_path}. Error {str(e)}")
        return ConfigBox({})
    
def create_directories(path_to_dir: List[Union[str, Path]], verbose: bool = True) -> None:
    """
    Creates directories specified in the list.
    
    Args:
        path_to_dir: List of directory paths to create
        verbose: If True, logs directory creation
    """
    for path in path_to_dir:
        # Convert to Path if string
        path_obj = Path(path) if isinstance(path, str) else path
        path_obj.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            logger.info(f"✅ Created directory: {path_obj}")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Saves a dictionary to a JSON file.
    
    Args:
        path: Path to the JSON file
        data: Data to be saved in the JSON file
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w", encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False, default=str)
        logger.info(f"✅ JSON file saved at {path}")
    except Exception as e:
        logger.error(f"❌ Error saving JSON to {path}: {e}")
        raise


def load_json(path: Path) -> ConfigBox:
    """
    Loads a dictionary from a JSON file.
    
    Args:
        path: Path to the JSON file
    
    Returns:
        ConfigBox: Data loaded from the JSON file
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        logger.error(f"❌ JSON file not found: {path}")
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        logger.info(f"✅ JSON file loaded from {path}")
        return ConfigBox(data)
    except Exception as e:
        logger.error(f"❌ Error loading JSON from {path}: {e}")
        raise
