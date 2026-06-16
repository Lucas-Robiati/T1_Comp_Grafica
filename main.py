#!/usr/bin/env python3
"""
main.py - Ponto de entrada do Sistema de Mapeamento de Objetos em Vídeo.

Inicializa o logging, carrega configurações e inicia a interface gráfica.
T1 - Computação Gráfica
"""

import os
import sys

# Garantir que o diretório raiz do projeto esteja no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from utils.config_loader import load_config
from gui.app import VideoMapperApp


def main():
    """Função principal: configura logging, carrega config e inicia a GUI."""
    # Configurar logger raiz
    root_logger = setup_logger("video_mapper", log_dir="logs")
    root_logger.info("=" * 60)
    root_logger.info("  Video Object Mapper — Iniciando aplicação")
    root_logger.info("=" * 60)

    # Carregar configurações
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config.yaml"
    )
    config = load_config(config_path)
    root_logger.info("Configurações carregadas com sucesso.")

    # Iniciar interface gráfica
    try:
        app = VideoMapperApp(config)
        app.run()
    except Exception as e:
        root_logger.error(f"Erro fatal na aplicação: {e}", exc_info=True)
        sys.exit(1)

    root_logger.info("Aplicação encerrada.")


if __name__ == "__main__":
    main()
