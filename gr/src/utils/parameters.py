import yaml


def process_parameters_yaml() -> dict:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    with open(f'parameters.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params


def get_yaml_parameter(name_parameter: str) -> dict:
    with open(f'parameters.yaml') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params[name_parameter]
