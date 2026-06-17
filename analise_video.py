"""
=============================================================================
 SISTEMA DE ANÁLISE DE VÍDEO - INTERFACE GRÁFICA (GUI)
=============================================================================
 Autor: Gerado para análise de movimento e rastreamento geométrico
 Descrição: Exibe o vídeo anotado, controles e painéis de estatísticas.
=============================================================================
"""

import os
import queue
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk

# Importa o processador de vídeo do módulo de processamento
from processamento import ProcessadorVideo


# =============================================================================
#  INTERFACE GRÁFICA — Thread principal
# =============================================================================

class AppAnaliseVideo:
    """
    Interface gráfica principal construída em Tkinter.
    Exibe o vídeo anotado, controles e painéis de estatísticas.
    """

    COR_FUNDO      = "#1e1e2e"
    COR_PAINEL     = "#2a2a3e"
    COR_ACENTO     = "#7c3aed"
    COR_SUCESSO    = "#10b981"
    COR_ALERTA     = "#f59e0b"
    COR_TEXTO      = "#e2e8f0"
    COR_SUBTEXTO   = "#94a3b8"
    COR_BOTAO_OK   = "#4f46e5"
    COR_BOTAO_STOP = "#dc2626"
    COR_BOTAO_PAU  = "#d97706"

    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.raiz.title("Análise de Vídeo — Visão Computacional Clássica")
        self.raiz.configure(bg=self.COR_FUNDO)
        self.raiz.resizable(True, True)

        # Fila de comunicação entre threads
        self.fila = queue.Queue(maxsize=5)

        # Instância do processador
        self.processador = ProcessadorVideo(self.fila)

        # Variáveis de estado da GUI
        self.caminho_video = tk.StringVar(value="Nenhum arquivo selecionado")
        self.var_threshold   = tk.IntVar(value=25)
        self.var_area        = tk.IntVar(value=100)
        self.var_persist     = tk.IntVar(value=30)
        self.var_confirmacao = tk.IntVar(value=5)
        self.var_progresso   = tk.DoubleVar(value=0.0)
        self.ultimo_frame_rgb = None

        self._construir_ui()
        self._poll_fila()   # Inicia verificação periódica da fila

    # ------------------------------------------------------------------
    #  Construção da Interface
    # ------------------------------------------------------------------

    def _construir_ui(self):
        """Monta todos os widgets da janela principal."""
        # ── Título ────────────────────────────────────────────────────
        topo = tk.Frame(self.raiz, bg=self.COR_FUNDO)
        topo.pack(fill="x", padx=15, pady=(12, 4))

        tk.Label(
            topo,
            text="🎬  Analisador de Vídeo — Visão Clássica",
            font=("Segoe UI", 16, "bold"),
            bg=self.COR_FUNDO,
            fg=self.COR_TEXTO
        ).pack(side="left")

        tk.Label(
            topo,
            text="OpenCV · MOG2 · Centroid + CSRT",
            font=("Segoe UI", 9),
            bg=self.COR_FUNDO,
            fg=self.COR_SUBTEXTO
        ).pack(side="left", padx=12)

        # ── Layout principal: vídeo + painel direito ───────────────────
        corpo = tk.Frame(self.raiz, bg=self.COR_FUNDO)
        corpo.pack(fill="both", expand=True, padx=15, pady=6)

        # Canvas do vídeo
        self.canvas = tk.Canvas(
            corpo,
            width=800, height=450,
            bg="#0f0f1a",
            highlightthickness=1,
            highlightbackground=self.COR_ACENTO
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._ao_redimensionar_canvas)

        # Painel direito (com Scrollbar para evitar corte de widgets)
        container_painel = tk.Frame(corpo, bg=self.COR_PAINEL, width=280)
        container_painel.pack(side="right", fill="y", padx=(10, 0))
        container_painel.pack_propagate(False)

        canvas_scroll = tk.Canvas(container_painel, bg=self.COR_PAINEL, highlightthickness=0)
        canvas_scroll.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container_painel, orient="vertical", command=canvas_scroll.yview)
        scrollbar.pack(side="right", fill="y")

        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        # Frame interno que realmente conterá os widgets
        painel = tk.Frame(canvas_scroll, bg=self.COR_PAINEL)
        canvas_window = canvas_scroll.create_window((0, 0), window=painel, anchor="nw")

        def _on_frame_configure(event):
            canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))

        def _on_canvas_configure(event):
            canvas_scroll.itemconfig(canvas_window, width=event.width)

        painel.bind("<Configure>", _on_frame_configure)
        canvas_scroll.bind("<Configure>", _on_canvas_configure)

        # Habilita scroll com a roda do mouse
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_up(event):
            canvas_scroll.yview_scroll(-1, "units")

        def _on_mousewheel_down(event):
            canvas_scroll.yview_scroll(1, "units")

        self._painel_arquivo(painel)
        self._painel_parametros(painel)
        self._painel_controles(painel)
        self._painel_estatisticas(painel)

        # Registra a rolagem de mouse de forma recursiva em todos os filhos
        def _registrar_scroll_recursivo(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel_up)
            widget.bind("<Button-5>", _on_mousewheel_down)
            for filho in widget.winfo_children():
                _registrar_scroll_recursivo(filho)

        _registrar_scroll_recursivo(canvas_scroll)
        _registrar_scroll_recursivo(painel)

        # ── Barra de progresso ─────────────────────────────────────────
        rodape = tk.Frame(self.raiz, bg=self.COR_FUNDO)
        rodape.pack(fill="x", padx=15, pady=(4, 10))

        tk.Label(
            rodape, text="Progresso:",
            bg=self.COR_FUNDO, fg=self.COR_SUBTEXTO,
            font=("Segoe UI", 8)
        ).pack(side="left")

        self.barra_prog = ttk.Progressbar(
            rodape,
            variable=self.var_progresso,
            maximum=100,
            length=650,
            mode="determinate"
        )
        self.barra_prog.pack(side="left", padx=8, fill="x", expand=True)

        self.lbl_pct = tk.Label(
            rodape, text="0%",
            bg=self.COR_FUNDO, fg=self.COR_TEXTO,
            font=("Segoe UI", 8, "bold"), width=5
        )
        self.lbl_pct.pack(side="left")

    def _painel_arquivo(self, pai):
        """Seção de seleção de arquivo."""
        frame = self._secao(pai, "📁  Arquivo de Vídeo")

        tk.Label(
            frame,
            textvariable=self.caminho_video,
            bg=self.COR_PAINEL,
            fg=self.COR_SUBTEXTO,
            font=("Segoe UI", 8),
            wraplength=230,
            justify="left"
        ).pack(anchor="w", pady=(0, 6))

        tk.Button(
            frame,
            text="Selecionar Vídeo",
            command=self._selecionar_video,
            bg=self.COR_BOTAO_OK, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=5
        ).pack(fill="x")

    def _painel_parametros(self, pai):
        """Seção de sliders de parâmetros."""
        frame = self._secao(pai, "⚙️  Parâmetros de Detecção")

        # Slider: Limiar de binarização
        self._slider(
            frame,
            label="Threshold VAR do MOG2",
            variavel=self.var_threshold,
            minval=10, maxval=100,
            callback=self._atualizar_threshold,
            dica="Menor = mais sensível; maior = mais conservador"
        )

        # Slider: Área mínima
        self._slider(
            frame,
            label="Área Mínima (px²)",
            variavel=self.var_area,
            minval=50, maxval=5000,
            callback=self._atualizar_area,
            dica="Filtra ruído por tamanho do contorno"
        )

        # Slider: Persistência do tracker
        self._slider(
            frame,
            label="Persistência (frames)",
            variavel=self.var_persist,
            minval=5, maxval=120,
            callback=self._atualizar_persist,
            dica="Frames sem detecção antes de esquecer objeto"
        )

        # Slider: Frames para confirmação CSRT
        self._slider(
            frame,
            label="Confirmação CSRT (frames)",
            variavel=self.var_confirmacao,
            minval=2, maxval=30,
            callback=self._atualizar_confirmacao,
            dica="Frames consecutivos p/ confirmar objeto"
        )

    def _painel_controles(self, pai):
        """Botões de controle: Iniciar / Pausar / Parar."""
        frame = self._secao(pai, "▶  Controles")

        self.btn_iniciar = tk.Button(
            frame,
            text="▶  Iniciar Processamento",
            command=self._iniciar,
            bg=self.COR_SUCESSO, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6
        )
        self.btn_iniciar.pack(fill="x", pady=(0, 4))

        self.btn_pausar = tk.Button(
            frame,
            text="⏸  Pausar",
            command=self._pausar,
            bg=self.COR_BOTAO_PAU, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6,
            state="disabled"
        )
        self.btn_pausar.pack(fill="x", pady=(0, 4))

        self.btn_parar = tk.Button(
            frame,
            text="⏹  Parar",
            command=self._parar,
            bg=self.COR_BOTAO_STOP, fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            padx=10, pady=6,
            state="disabled"
        )
        self.btn_parar.pack(fill="x")

    def _painel_estatisticas(self, pai):
        """Painel de estatísticas em tempo real."""
        frame = self._secao(pai, "📊  Estatísticas em Tempo Real")

        self.stat_frames   = self._stat_linha(frame, "Frames com movimento:", "0")
        self.stat_objetos  = self._stat_linha(frame, "Confirmados (CSRT):",   "0")
        self.stat_ativos   = self._stat_linha(frame, "Objetos ativos agora:", "0")
        self.stat_status   = self._stat_linha(frame, "Status:",               "Aguardando")

    # ------------------------------------------------------------------
    #  Helpers de layout
    # ------------------------------------------------------------------

    def _secao(self, pai, titulo: str) -> tk.Frame:
        """Cria uma seção com título dentro do painel."""
        wrapper = tk.Frame(pai, bg=self.COR_PAINEL)
        wrapper.pack(fill="x", padx=10, pady=(10, 2))

        tk.Label(
            wrapper,
            text=titulo,
            bg=self.COR_PAINEL,
            fg=self.COR_ACENTO,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(0, 4))

        sep = tk.Frame(wrapper, bg=self.COR_ACENTO, height=1)
        sep.pack(fill="x", pady=(0, 6))

        return wrapper

    def _slider(self, pai, label, variavel, minval, maxval, callback, dica=""):
        """Cria um slider com rótulo e exibição de valor de forma compacta."""
        row = tk.Frame(pai, bg=self.COR_PAINEL)
        row.pack(fill="x", pady=4)

        # Linha do topo: Rótulo à esquerda, Valor em destaque à direita
        topo_linha = tk.Frame(row, bg=self.COR_PAINEL)
        topo_linha.pack(fill="x")

        tk.Label(
            topo_linha, text=label,
            bg=self.COR_PAINEL, fg=self.COR_TEXTO,
            font=("Segoe UI", 8, "bold")
        ).pack(side="left", anchor="w")

        lbl_val = tk.Label(
            topo_linha, textvariable=variavel,
            bg=self.COR_PAINEL, fg=self.COR_ACENTO,
            font=("Segoe UI", 8, "bold")
        ).pack(side="right", anchor="e")

        if dica:
            tk.Label(
                row, text=dica,
                bg=self.COR_PAINEL, fg=self.COR_SUBTEXTO,
                font=("Segoe UI", 7)
            ).pack(anchor="w", pady=(0, 2))

        sl = tk.Scale(
            row,
            variable=variavel,
            from_=minval, to=maxval,
            orient="horizontal",
            bg=self.COR_PAINEL, fg=self.COR_TEXTO,
            troughcolor="#3b3b5c",
            highlightthickness=0,
            showvalue=False,
            command=callback
        )
        sl.pack(fill="x", expand=True)

    def _stat_linha(self, pai, rotulo: str, valor_inicial: str) -> tk.Label:
        """Cria uma linha de estatística com rótulo e valor dinâmico."""
        row = tk.Frame(pai, bg=self.COR_PAINEL)
        row.pack(fill="x", pady=2)

        tk.Label(
            row, text=rotulo,
            bg=self.COR_PAINEL, fg=self.COR_SUBTEXTO,
            font=("Segoe UI", 8)
        ).pack(side="left")

        lbl = tk.Label(
            row, text=valor_inicial,
            bg=self.COR_PAINEL, fg=self.COR_SUCESSO,
            font=("Segoe UI", 8, "bold")
        )
        lbl.pack(side="right")
        return lbl

    # ------------------------------------------------------------------
    #  Callbacks dos controles
    # ------------------------------------------------------------------

    def _selecionar_video(self):
        """
        Abre o gerenciador de arquivos nativo do sistema via zenity/kdialog.
        Fallback para tkinter.filedialog caso não esteja disponível.
        """
        caminho = None

        # Tenta zenity (GNOME/GTK)
        try:
            resultado = subprocess.run(
                [
                    "zenity", "--file-selection",
                    "--title=Selecionar Vídeo",
                    "--file-filter=Vídeos (mp4 avi mkv mov) | *.mp4 *.avi *.mkv *.mov *.wmv *.flv",
                    "--file-filter=Todos os arquivos | *"
                ],
                capture_output=True, text=True, timeout=120
            )
            if resultado.returncode == 0 and resultado.stdout.strip():
                caminho = resultado.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Tenta kdialog (KDE) se zenity falhou
        if caminho is None:
            try:
                resultado = subprocess.run(
                    [
                        "kdialog", "--getopenfilename",
                        os.path.expanduser("~"),
                        "Vídeos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv)"
                    ],
                    capture_output=True, text=True, timeout=120
                )
                if resultado.returncode == 0 and resultado.stdout.strip():
                    caminho = resultado.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Fallback: tkinter filedialog
        if caminho is None:
            caminho = filedialog.askopenfilename(
                title="Selecionar Vídeo",
                filetypes=[
                    ("Arquivos de Vídeo", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
                    ("Todos os arquivos", "*.*")
                ]
            )

        if caminho:
            self.caminho_video.set(os.path.basename(caminho))
            self._caminho_completo = caminho

    def _iniciar(self):
        """Inicia o processamento do vídeo selecionado."""
        if not hasattr(self, "_caminho_completo") or not self._caminho_completo:
            messagebox.showwarning("Atenção", "Selecione um arquivo de vídeo primeiro.")
            return

        if self.processador.rodando:
            messagebox.showinfo("Info", "Já há um processamento em andamento.")
            return

        # Aplica parâmetros atuais
        self.processador.var_threshold = self.var_threshold.get()
        self.processador.area_minima        = self.var_area.get()
        self.processador.max_desaparecido   = self.var_persist.get()
        self.processador.tracker.max_desaparecido = self.var_persist.get()
        self.processador.frames_confirmacao = self.var_confirmacao.get()

        # Atualiza botões
        self.btn_iniciar.config(state="disabled")
        self.btn_pausar.config(state="normal", text="⏸  Pausar")
        self.btn_parar.config(state="normal")
        self.stat_status.config(text="Processando...", fg=self.COR_ALERTA)
        self.var_progresso.set(0)
        self.canvas.delete("placeholder")

        self.processador.iniciar(self._caminho_completo)

    def _pausar(self):
        """Alterna pausa/retomada."""
        if not self.processador.rodando:
            return
        self.processador.pausar()
        if self.processador.pausado:
            self.btn_pausar.config(text="▶  Retomar")
            self.stat_status.config(text="Pausado", fg=self.COR_ALERTA)
        else:
            self.btn_pausar.config(text="⏸  Pausar")
            self.stat_status.config(text="Processando...", fg=self.COR_ALERTA)

    def _parar(self):
        """Para o processamento."""
        self.processador.parar()
        self.btn_iniciar.config(state="normal")
        self.btn_pausar.config(state="disabled", text="⏸  Pausar")
        self.btn_parar.config(state="disabled")
        self.stat_status.config(text="Interrompido", fg=self.COR_BOTAO_STOP)

    def _atualizar_threshold(self, val):
        self.processador.var_threshold = int(val)

    def _atualizar_area(self, val):
        self.processador.area_minima = int(val)

    def _atualizar_persist(self, val):
        v = int(val)
        self.processador.max_desaparecido = v
        self.processador.tracker.max_desaparecido = v

    def _atualizar_confirmacao(self, val):
        self.processador.frames_confirmacao = int(val)

    # ------------------------------------------------------------------
    #  Poll da fila (atualização da GUI na thread principal)
    # ------------------------------------------------------------------

    def _poll_fila(self):
        """
        Verifica periodicamente se há mensagens da thread de processamento.
        Deve ser chamado somente na thread principal do Tkinter.
        """
        try:
            while True:
                msg = self.fila.get_nowait()
                tipo = msg[0]

                if tipo == "frame":
                    _, frame_rgb, progresso, n_ativos, n_frames, n_total = msg
                    self._exibir_frame(frame_rgb)
                    self.var_progresso.set(progresso)
                    self.lbl_pct.config(text=f"{progresso}%")
                    self.stat_frames.config(text=str(n_frames))
                    self.stat_objetos.config(text=str(n_total))
                    self.stat_ativos.config(text=str(n_ativos))

                elif tipo == "concluido":
                    _, n_frames, n_total = msg
                    self.btn_iniciar.config(state="normal")
                    self.btn_pausar.config(state="disabled", text="⏸  Pausar")
                    self.btn_parar.config(state="disabled")
                    self.var_progresso.set(100)
                    self.lbl_pct.config(text="100%")
                    self.stat_status.config(text="Concluído ✓", fg=self.COR_SUCESSO)
                    messagebox.showinfo(
                        "Processamento Concluído",
                        f"Análise finalizada!\n\n"
                        f"• Frames com movimento salvos: {n_frames}\n"
                        f"  → Pasta: saida/frames/\n\n"
                        f"• Objetos confirmados (CSRT): {n_total}\n"
                        f"  → Pasta: saida/objetos/\n\n"
                        f"ℹ Objetos que duraram < {self.var_confirmacao.get()} frames\n"
                        f"  foram descartados como falsos positivos."
                    )

                elif tipo == "erro":
                    messagebox.showerror("Erro", msg[1])
                    self.btn_iniciar.config(state="normal")
                    self.btn_pausar.config(state="disabled")
                    self.btn_parar.config(state="disabled")

        except queue.Empty:
            pass

        # Reagenda a cada 30ms (~33 fps para a GUI)
        self.raiz.after(30, self._poll_fila)

    def _exibir_frame(self, frame_rgb: np.ndarray):
        """Converte um array NumPy RGB em imagem Tkinter e exibe no canvas, mantendo a proporção."""
        self.ultimo_frame_rgb = frame_rgb
        img_pil = Image.fromarray(frame_rgb)

        # Ajusta ao tamanho atual do canvas mantendo proporção (Aspect Ratio)
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 450

        largura_original, altura_original = img_pil.size
        proporcao_img = largura_original / altura_original
        proporcao_canvas = cw / ch

        if proporcao_canvas > proporcao_img:
            # Canvas é mais largo que a imagem (barra preta nas laterais)
            nova_altura = ch
            nova_largura = int(ch * proporcao_img)
        else:
            # Canvas é mais alto que a imagem (barra preta no topo/baixo)
            nova_largura = cw
            nova_altura = int(cw / proporcao_img)

        # Redimensiona mantendo qualidade e proporção (valores mínimos de 1 para evitar crash de tamanho 0)
        img_pil = img_pil.resize((max(nova_largura, 1), max(nova_altura, 1)), Image.LANCZOS)

        self._img_tk = ImageTk.PhotoImage(img_pil)
        
        # Centraliza a imagem no Canvas
        x_pos = (cw - nova_largura) // 2
        y_pos = (ch - nova_altura) // 2

        self.canvas.delete("all")
        self.canvas.create_image(x_pos, y_pos, anchor="nw", image=self._img_tk)

    def _ao_redimensionar_canvas(self, event):
        """Manipula o redimensionamento do canvas para atualizar o frame ou centralizar o placeholder."""
        if hasattr(self, "ultimo_frame_rgb") and self.ultimo_frame_rgb is not None:
            self._exibir_frame(self.ultimo_frame_rgb)
        else:
            cw = event.width
            ch = event.height
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Selecione um arquivo de vídeo para começar",
                fill=self.COR_SUBTEXTO,
                font=("Segoe UI", 13),
                tags="placeholder"
            )


# =============================================================================
#  PONTO DE ENTRADA
# =============================================================================

def main():
    raiz = tk.Tk()
    raiz.geometry("1150x640")
    raiz.minsize(950, 550)

    # Estilo da barra de progresso via ttk
    estilo = ttk.Style()
    estilo.theme_use("clam")
    estilo.configure(
        "TProgressbar",
        troughcolor="#2a2a3e",
        background="#7c3aed",
        thickness=14
    )

    app = AppAnaliseVideo(raiz)
    raiz.protocol("WM_DELETE_WINDOW", lambda: (app.processador.parar(), raiz.destroy()))
    raiz.mainloop()


if __name__ == "__main__":
    main()
