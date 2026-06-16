"""
logger.py - Configuração centralizada do sistema de logging.

Fornece um setup padronizado com handlers para console e arquivo,
com formatação consistente em todos os módulos do sistema.
"""

import logging
import os
from datetime import datetime


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configura e retorna um logger com handlers para console e arquivo.

    Args:
        name: Nome do logger (geralmente __name__ do módulo).
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Diretório onde salvar os arquivos de log.

    Returns:
        Logger configurado com handlers para console e arquivo.
    """
    logger = logging.getLogger(name)

    # Evita adicionar handlers duplicados se o logger já existir
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formato padronizado: [timestamp] [level] [module] message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)-20s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para console (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(
            log_dir,
            f"video_mapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Arquivo captura tudo
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"Não foi possível criar arquivo de log: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger existente ou cria um novo com configuração padrão.

    Args:
        name: Nome do logger.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
