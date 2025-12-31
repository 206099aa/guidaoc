import yaml
import os
import re
import itertools
import copy
import logging

logger = logging.getLogger("ConfigMgr")


class UnitParser:
    """SI Unit Converter for Physics Engine"""
    UNITS = {
        'km/h': 1 / 3.6, 'm/s': 1.0,
        'kg': 1.0, 'ton': 1000.0,
        'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.0,
        'min': 60.0, 'h': 3600.0, 's': 1.0
    }

    @staticmethod
    def parse(val):
        if isinstance(val, str):
            match = re.match(r'^([\d\.]+)\s*([a-zA-Z/]+)$', val.strip())
            if match:
                num, unit = match.groups()
                if unit in UnitParser.UNITS:
                    return float(num) * UnitParser.UNITS[unit]
        return val


class ConfigLoader:
    """
    [Configuration]
    Handles parameter injection for Sensitivity Analysis (Essential for SCI).
    """

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config {path} not found")
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        return ConfigLoader._recursive_parse(raw)

    @staticmethod
    def _recursive_parse(data):
        if isinstance(data, dict):
            return {k: ConfigLoader._recursive_parse(v) for k, v in data.items()}
        return UnitParser.parse(data)

    @staticmethod
    def generate_sweep(base_path, sweep_params):
        """
        Generate a sequence of configs for batch experimentation.
        Example sweep: {'environment.mud_factor': [0.1, 0.5, 0.9]}
        """
        base = ConfigLoader.load(base_path)
        keys, values = zip(*sweep_params.items())

        for bundle in itertools.product(*values):
            cfg = copy.deepcopy(base)
            name_suffix = []

            for k, v in zip(keys, bundle):
                # Update nested key
                ref = cfg
                path = k.split('.')
                for p in path[:-1]: ref = ref.setdefault(p, {})
                ref[path[-1]] = v
                name_suffix.append(f"{path[-1]}_{v}")

            cfg['meta']['experiment_id'] += "_" + "_".join(name_suffix)
            yield cfg