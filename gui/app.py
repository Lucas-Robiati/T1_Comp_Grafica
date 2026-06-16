"""
app.py - Interface gráfica principal (Tkinter).

Fornece a janela principal com seleção de arquivo, barra de progresso,
log em tempo real, botões de controle e resumo final.
"""

import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional

from engine.video_processor import VideoProcessor
from plugins.twilio_notifier import TwilioNotifier

logger = logging.getLogger(__name__)


class VideoMapperApp:
    """
    Aplicação Tkinter para mapeamento de objetos em vídeo.

    A GUI roda na main thread. O processamento pesado roda em worker thread.
    Comunicação via queue.Queue + root.after() polling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dicionário de configurações carregado pelo config_loader.
        """
        self.config = config
        self.root = tk.Tk()
        self.root.title("Video Object Mapper — T1 Computação Gráfica")
        self.root.geometry("900x700")
        self.root.minsize(750, 550)
        self.root.configure(bg="#1e1e2e")

        # Estado interno
        self.video_path: Optional[str] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.log_queue: queue.Queue = queue.Queue()
        self.progress_value = tk.DoubleVar(value=0.0)
        self.is_processing = False

        # Montar interface
        self._setup_styles()
        self._build_gui()

        # Iniciar polling de logs
        self._poll_log_queue()

        logger.info("GUI inicializada.")

    def _setup_styles(self):
        """Configura estilos visuais do ttk."""
        style = ttk.Style()
        style.theme_use("clam")

        # Cores
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg,
                         font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=bg, foreground=accent,
                         font=("Segoe UI", 16, "bold"))
        style.configure("TButton", font=("Segoe UI", 11), padding=8)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"),
                         padding=10)
        style.configure("Horizontal.TProgressbar", troughcolor=surface,
                         background=accent, thickness=22)

    def _build_gui(self):
        """Constrói todos os widgets da interface."""
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        surface = "#313244"

        # --- Frame principal ---
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Título ---
        title_label = ttk.Label(
            main_frame,
            text="🎬 Video Object Mapper",
            style="Title.TLabel"
        )
        title_label.pack(pady=(0, 15))

        # --- Frame de seleção de arquivo ---
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Label(file_frame, text="Arquivo de vídeo:").pack(
            side=tk.LEFT, padx=(0, 10)
        )

        self.file_entry = tk.Entry(
            file_frame, font=("Segoe UI", 10), bg=surface,
            fg=fg, insertbackground=fg, relief=tk.FLAT,
            highlightthickness=1, highlightcolor="#89b4fa"
        )
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.browse_btn = ttk.Button(
            file_frame, text="Selecionar...", command=self._browse_file
        )
        self.browse_btn.pack(side=tk.RIGHT)

        # --- Barra de progresso ---
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)

        ttk.Label(progress_frame, text="Progresso:").pack(side=tk.LEFT, padx=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_value,
            maximum=100, mode="determinate",
            style="Horizontal.TProgressbar"
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.progress_label = ttk.Label(progress_frame, text="0%", width=6)
        self.progress_label.pack(side=tk.RIGHT)

        # --- Log area ---
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(log_frame, text="Log de Processamento:").pack(
            anchor=tk.W, pady=(0, 5)
        )

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=18, font=("Consolas", 10),
            bg=surface, fg=fg, insertbackground=fg,
            relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED,
            highlightthickness=1, highlightcolor="#89b4fa"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Botões de ação ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.process_btn = ttk.Button(
            btn_frame, text="▶  Processar Vídeo",
            style="Accent.TButton", command=self._start_processing
        )
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.cancel_btn = ttk.Button(
            btn_frame, text="⏹  Cancelar",
            command=self._cancel_processing, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT)

        # --- Info Twilio ---
        twilio_cfg = self.config.get("twilio", {})
        twilio_ok = all([
            twilio_cfg.get("account_sid"), twilio_cfg.get("auth_token"),
            twilio_cfg.get("from_phone"), twilio_cfg.get("to_phone"),
        ])
        twilio_status = "✅ Twilio configurado" if twilio_ok else "⚠️ Twilio não configurado (SMS desabilitado)"
        twilio_label = ttk.Label(btn_frame, text=twilio_status,
                                  font=("Segoe UI", 9))
        twilio_label.pack(side=tk.RIGHT)

    def _browse_file(self):
        """Abre diálogo para selecionar arquivo de vídeo."""
        filetypes = [
            ("Vídeos", "*.mp4 *.avi *.mov *.mkv"),
            ("MP4", "*.mp4"),
            ("AVI", "*.avi"),
            ("MOV", "*.mov"),
            ("Todos", "*.*"),
        ]
        path = filedialog.askopenfilename(
            title="Selecionar arquivo de vídeo",
            filetypes=filetypes
        )
        if path:
            self.video_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)
            self._append_log(f"Arquivo selecionado: {os.path.basename(path)}")

    def _append_log(self, message: str):
        """Adiciona mensagem ao widget de log (thread-safe via queue)."""
        self.log_queue.put(message)

    def _poll_log_queue(self):
        """Polling para consumir mensagens da queue e atualizar a GUI."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass

        # Atualizar label de progresso
        pv = self.progress_value.get()
        self.progress_label.configure(text=f"{pv:.1f}%")

        # Agendar próximo polling
        self.root.after(100, self._poll_log_queue)

    def _start_processing(self):
        """Inicia o processamento do vídeo em thread separada."""
        if self.is_processing:
            messagebox.showwarning("Aviso", "Processamento já em andamento.")
            return

        path = self.file_entry.get().strip()
        if not path:
            messagebox.showwarning("Aviso", "Selecione um arquivo de vídeo primeiro.")
            return

        if not os.path.exists(path):
            messagebox.showerror("Erro", f"Arquivo não encontrado:\n{path}")
            return

        self.video_path = path
        self.is_processing = True
        self.cancel_event.clear()
        self.progress_value.set(0.0)

        # Limpar log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        # Atualizar estado dos botões
        self.process_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.NORMAL)
        self.browse_btn.configure(state=tk.DISABLED)

        self._append_log("=" * 50)
        self._append_log("  Iniciando processamento de vídeo")
        self._append_log("=" * 50)

        # Criar processador
        processor = VideoProcessor(
            config=self.config,
            cancel_event=self.cancel_event,
            progress_callback=lambda v: self.progress_value.set(v),
            log_callback=self._append_log,
        )

        # Iniciar thread de processamento
        def worker():
            result = processor.process(self.video_path)
            self.root.after(0, lambda: self._on_processing_complete(result))

        self.processing_thread = threading.Thread(target=worker, daemon=True)
        self.processing_thread.start()

    def _cancel_processing(self):
        """Sinaliza cancelamento do processamento."""
        if self.is_processing:
            self.cancel_event.set()
            self._append_log("Solicitação de cancelamento enviada...")

    def _on_processing_complete(self, result: Dict[str, Any]):
        """Callback chamado quando o processamento termina."""
        self.is_processing = False
        self.process_btn.configure(state=tk.NORMAL)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.browse_btn.configure(state=tk.NORMAL)

        status = result.get("status", "error")

        if status == "success":
            self._append_log("\n" + "=" * 50)
            self._append_log("  PROCESSAMENTO CONCLUÍDO COM SUCESSO")
            self._append_log("=" * 50)

            # Montar resumo
            total = result.get("total_unique_objects", 0)
            top = result.get("top_classes", [])
            files = result.get("output_files", [])

            top_str = ", ".join([f"{c} ({n})" for c, n in top[:3]])
            files_str = "\n".join([f"  • {f}" for f in files])

            summary = (
                f"Objetos únicos detectados: {total}\n"
                f"Classes predominantes: {top_str}\n\n"
                f"Arquivos de saída:\n{files_str}"
            )
            messagebox.showinfo("Processamento Concluído", summary)

            # Enviar SMS via Twilio
            notifier = TwilioNotifier(self.config)
            notifier.send_notification(
                total_objects=total,
                top_classes=top,
                status="sucesso",
                callback=lambda ok, msg: self._append_log(
                    f"Twilio: {msg}"
                ),
            )

        elif status == "cancelled":
            self._append_log("Processamento foi cancelado.")
            messagebox.showinfo("Cancelado", "Processamento cancelado pelo usuário.")

        else:
            error_msg = result.get("error_message", "Erro desconhecido")
            self._append_log(f"Processamento falhou: {error_msg}")
            messagebox.showerror("Erro", f"Processamento falhou:\n{error_msg}")

            # Notificar falha via Twilio
            notifier = TwilioNotifier(self.config)
            notifier.send_notification(
                total_objects=0, top_classes=[], status="falha",
                callback=lambda ok, msg: self._append_log(f"Twilio: {msg}"),
            )

    def run(self):
        """Inicia o loop principal da interface gráfica."""
        logger.info("Iniciando mainloop da GUI.")
        self.root.mainloop()
