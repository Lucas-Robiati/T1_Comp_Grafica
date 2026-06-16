"""
twilio_notifier.py - Plugin de notificação SMS via API Twilio.

Envia SMS ao término do processamento com resumo dos resultados.
Roda em thread separada para não bloquear a GUI.
Se as credenciais não estiverem configuradas, exibe aviso e prossegue.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TwilioNotifier:
    """
    Notificador SMS via API Twilio.

    Lê credenciais do config. Se ausentes, opera em modo desabilitado
    sem causar erros.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dicionário completo de configuração.
        """
        twilio_cfg = config.get("twilio", {})
        self.account_sid = twilio_cfg.get("account_sid", "")
        self.auth_token = twilio_cfg.get("auth_token", "")
        self.from_phone = twilio_cfg.get("from_phone", "")
        self.to_phone = twilio_cfg.get("to_phone", "")

        # Verificar se as credenciais estão completas
        self.enabled = all([
            self.account_sid, self.auth_token,
            self.from_phone, self.to_phone
        ])

        if self.enabled:
            logger.info("TwilioNotifier habilitado e configurado.")
        else:
            logger.info(
                "TwilioNotifier desabilitado: credenciais Twilio ausentes "
                "no config.yaml ou variáveis de ambiente."
            )

    def _format_message(
        self,
        total_objects: int,
        top_classes: List[Tuple[str, int]],
        status: str,
    ) -> str:
        """
        Formata a mensagem SMS com o resumo do processamento.

        Args:
            total_objects: Número total de objetos únicos detectados.
            top_classes: Lista de tuplas (classe, contagem) top 3.
            status: "sucesso" ou "falha".

        Returns:
            Texto formatado para o SMS.
        """
        top_str = ", ".join(
            [f"{cls} ({count})" for cls, count in top_classes[:3]]
        )
        message = (
            f"[Video Object Mapper] Processamento: {status.upper()}\n"
            f"Objetos únicos detectados: {total_objects}\n"
            f"Top 3 classes: {top_str}\n"
        )
        return message

    def send_notification(
        self,
        total_objects: int,
        top_classes: List[Tuple[str, int]],
        status: str,
        callback: Optional[Callable[[bool, str], None]] = None,
    ):
        """
        Envia notificação SMS de forma assíncrona.

        Args:
            total_objects: Número total de objetos únicos.
            top_classes: Top 3 classes detectadas.
            status: Status do processamento.
            callback: Função chamada ao completar (success: bool, msg: str).
        """
        if not self.enabled:
            logger.info("SMS não enviado: Twilio não está configurado.")
            if callback:
                callback(False, "Twilio não configurado.")
            return

        def _send():
            try:
                from twilio.rest import Client

                client = Client(self.account_sid, self.auth_token)
                body = self._format_message(total_objects, top_classes, status)

                message = client.messages.create(
                    body=body,
                    from_=self.from_phone,
                    to=self.to_phone,
                )

                logger.info(f"SMS enviado com sucesso. SID: {message.sid}")
                if callback:
                    callback(True, f"SMS enviado (SID: {message.sid})")

            except ImportError:
                msg = ("Biblioteca 'twilio' não instalada. "
                       "Instale com: pip install twilio")
                logger.warning(msg)
                if callback:
                    callback(False, msg)

            except Exception as e:
                msg = f"Erro ao enviar SMS: {e}"
                logger.error(msg)
                if callback:
                    callback(False, msg)

        # Executar em thread separada
        thread = threading.Thread(target=_send, daemon=True)
        thread.start()
        logger.debug("Thread de envio SMS iniciada.")
