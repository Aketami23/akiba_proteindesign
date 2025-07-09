import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        _config = yaml.safe_load(file)
    return _config