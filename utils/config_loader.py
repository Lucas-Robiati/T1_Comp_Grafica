"""
config_loader.py - Carregamento e validação de configurações.

Suporta config.yaml como fonte principal e .env para variáveis sensíveis
(credenciais Twilio). Fornece valores padrão seguros para todos os parâmetros.
"""

import os
import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Valores padrão caso config.yaml não exista ou esteja incompleto
DEFAULT_CONFIG: Dict[str, Any] = {
    "processing": {
        "frame_interval_seconds": 2,
        "confidence_threshold": 0.4,
        "model_name": "yolov8n.pt",
        "iou_threshold": 0.3,
        "max_age": 10,
    },
    "twilio": {
        "account_sid": "",
        "auth_token": "",
        "from_phone": "",
        "to_phone": "",
    },
    "output": {
        "directory": "output",
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Faz merge profundo de dois dicionários. O 'override' sobrescreve o 'base'.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Carrega configurações do arquivo YAML com fallback para valores padrão.

    Também verifica variáveis de ambiente para credenciais Twilio:
        - TWILIO_ACCOUNT_SID
        - TWILIO_AUTH_TOKEN
        - TWILIO_FROM_PHONE
        - TWILIO_TO_PHONE

    Args:
        config_path: Caminho para o arquivo config.yaml.

    Returns:
        Dicionário com todas as configurações.
    """
    config = DEFAULT_CONFIG.copy()

    # Tenta carregar config.yaml
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
            if file_config and isinstance(file_config, dict):
                config = _deep_merge(config, file_config)
                logger.info(f"Configurações carregadas de '{config_path}'")
            else:
                logger.warning(
                    f"Arquivo '{config_path}' está vazio ou inválido. "
                    "Usando valores padrão."
                )
        except yaml.YAMLError as e:
            logger.error(f"Erro ao parsear '{config_path}': {e}")
        except OSError as e:
            logger.error(f"Erro ao ler '{config_path}': {e}")
    else:
        logger.info(
            f"Arquivo '{config_path}' não encontrado. Usando valores padrão."
        )

    # Tenta carregar .env (se python-dotenv estiver disponível)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Variáveis de ambiente sobrescrevem config.yaml para credenciais Twilio
    env_mappings = {
        "TWILIO_ACCOUNT_SID": ("twilio", "account_sid"),
        "TWILIO_AUTH_TOKEN": ("twilio", "auth_token"),
        "TWILIO_FROM_PHONE": ("twilio", "from_phone"),
        "TWILIO_TO_PHONE": ("twilio", "to_phone"),
    }

    for env_var, (section, key) in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value:
            config[section][key] = env_value
            logger.debug(f"Credencial '{env_var}' carregada do ambiente.")

    return config
