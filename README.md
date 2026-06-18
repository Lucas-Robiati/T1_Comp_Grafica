# 🎬 Analisador de Vídeo — OpenCV

Sistema de detecção e rastreamento de objetos em vídeo

---

## 🛠️ Tecnologias e Técnicas Empregadas

O pipeline atual resolve problemas crônicos de ruído e fragmentação através da combinação inteligente das seguintes técnicas de Visão Computacional:

- **Pré-processamento Adaptativo (CLAHE):** Equalização local de histograma para melhorar contraste antes da segmentação.
- **Segmentação de Fundo (MOG2):** Extração de objetos em movimento com modelo estatístico temporal.
- **Filtros Geométricos e Morfológicos:** Eliminação rígida de ruídos avaliando `Área`, `Razão de Aspecto`, `Extensão` e `Solidez` do contorno.
- **NMS (Non-Maximum Suppression):** Supressão de caixas delimitadoras sobrepostas para evitar múltiplas detecções da mesma parte do objeto.
- **Rastreamento Húngaro + Filtro de Kalman:** Predição de velocidade e posição dos objetos lidando com oclusões parciais, perda temporária de detecção e cruzamentos.
- **CSRT (Discriminative Correlation Filter):** Atua **estritamente como validador**. Um objeto só é confirmado como real se persistir estavelmente por $N$ frames no rastreamento CSRT, filtrando reflexos ou movimentos falsos rápidos.
- **Deduplicação por Aparência:** Cálculo de histograma de cores em espaço HSV (comparado via distância de Bhattacharyya) para evitar salvar imagens duplicadas de um mesmo objeto que foi re-identificado.
- **Seleção Dinâmica por Nitidez:** Escolhe o "melhor" recorte de um objeto ativo baseando-se no cálculo da variância do Laplaciano (evitando borrões de movimento).

---

## 🚀 Instalação e Execução

### 1. Dependências
```bash
pip install -r requirements.txt
```

> **Nota sobre Interface (Tkinter):**
> O módulo `tkinter` é utilizado para a interface gráfica e geralmente já é nativo na instalação Python. Caso acuse falta do módulo no Linux/macOS:
> - Ubuntu/Debian: `sudo apt-get install python3-tk`
> - macOS (Homebrew): `brew install python-tk`

### 2. Rodando o App
```bash
python analise_video.py
```

---

## 📂 Estrutura de Saídas

Os resultados são gerados dentro da pasta `saida/`, recriada automaticamente a cada execução:

```
saida/
├── frames/
│   ├── frame_1.jpg      ← Frame completo onde há objetos confirmados no momento
│   ├── frame_2.jpg
│   └── ...
└── objetos/
    ├── objeto_id_1.jpg  ← Melhor recorte (mais nítido e não duplicado) do Objeto 1
    ├── objeto_id_2.jpg
    └── ...
```

---

## ⚙️ Arquitetura do Pipeline de Processamento (Por Frame)

1. **Leitura e CLAHE**: Frame lido, redimensionado para 800x450 e contraste equalizado.
2. **Subtração de Fundo (MOG2)**: Detecta movimento, ignorando sombras.
3. **Filtro de Contornos**: Remove o que for muito pequeno, longo demais ou de bordas irreais (Solidez/Extensão baixa).
4. **NMS por IoU**: Apaga caixas que se sobrepõem no mesmo lugar (evitando objetos fantasmas sobre os reais).
5. **Kalman + Húngaro**: Associa a caixa a um ID existente, ou cria um novo com cooldown espacial (não cadastra em cima de outro ativo).
6. **Validação CSRT**: Inicializa sub-rastreadores; o objeto ganha o status verde apenas após se manter visível e coerente por uma sequência de frames.
7. **Avaliação Laplaciana e Histograma**: Separa o frame mais focado do objeto. Compara a cor; se for "gêmeo" de outro já salvo, a gravação é ignorada.
8. **GUI Assíncrona**: O frame desenhado e as estatísticas são despachados para a thread do Tkinter sem travar a interface.

---

## 🎛️ Controles e Parâmetros da Interface

Na aba da direita da aplicação, você tem controle dinâmico dos principais parâmetros antiruído:

| Parâmetro | Descrição |
|-----------|-----------|
| **Threshold VAR do MOG2** | Quão drástica a mudança no pixel precisa ser para ser movimento (maior = mais conservador / menos ruído). |
| **Área Mínima (px²)** | Ignora poeira e objetos irrelevantes pelo tamanho total. |
| **Persistência (frames)** | Tempo de sobrevida de um objeto antes do KalmanTracker "esquecê-lo". Útil quando ele passa por trás de um poste. |
| **Confirmação CSRT** | A barreira antiruído rápido. Quantos frames seguidos ele precisa "existir de fato" antes de disparar gatilhos e salvamentos. |
