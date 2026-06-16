# 🎬 Analisador de Vídeo — Visão Computacional Clássica

Software de análise de vídeo **sem IA** — detecta movimento e rastreia objetos usando apenas OpenCV clássico, morfologia matemática e geometria euclidiana.

---

## Instalação

```bash
pip install -r requirements.txt
```

> **tkinter** é nativo do Python. Se não disponível:
> - Ubuntu/Debian: `sudo apt-get install python3-tk`
> - macOS (Homebrew): `brew install python-tk`

---

## Execução

```bash
python analise_video.py
```

---

## Estrutura de Saídas

```
saida/
├── frames/
│   ├── frame_1.jpg      ← frame com movimento detectado
│   ├── frame_2.jpg
│   └── ...
└── objetos/
    ├── objeto_id_1.jpg  ← recorte do 1º objeto ao aparecer
    ├── objeto_id_2.jpg
    └── ...
```

> ⚠️ As pastas são **apagadas e recriadas** a cada novo processamento.

---

## Arquitetura Técnica

### Pipeline de Detecção (por frame)

```
Frame bruto
    ↓ Redimensionamento (800×450)
    ↓ Escala de cinza + Gaussian Blur
    ↓ MOG2 Background Subtractor
    ↓ Threshold binário (limiar configurável)
    ↓ Morfologia: OPEN → CLOSE → Dilate
    ↓ findContours → filtra por área mínima
    ↓
    ├─ Filtro 1: salva frame em saida/frames/
    └─ Filtro 2: Centroid Tracker → salva recortes em saida/objetos/
```

### Centroid Tracker (Rastreamento Clássico)

| Conceito | Implementação |
|----------|---------------|
| Identificação | ID inteiro único e sequencial (nunca reutilizado) |
| Associação | Distância euclidiana mínima via `scipy.spatial.distance.cdist` |
| Persistência | Contador de frames ausentes por objeto |
| Descarte | Objeto removido após `max_desaparecido` frames sem detecção |
| Salvamento | Recorte salvo **uma única vez** por ID, ao primeiro aparecimento |

---

## Parâmetros da GUI

| Parâmetro | Padrão | Efeito |
|-----------|--------|--------|
| **Limiar de Binarização** | 25 | Menor = mais sensível ao movimento |
| **Área Mínima (px²)** | 500 | Filtra ruído pequeno (pixels isolados) |
| **Persistência (frames)** | 30 | Quantos frames o objeto pode sumir sem ser esquecido |

---

## Tecnologias Utilizadas

- `cv2.createBackgroundSubtractorMOG2` — subtração de fundo estatística (Mistura de Gaussianas)
- `cv2.morphologyEx` — limpeza morfológica (OPEN, CLOSE, DILATE)
- `cv2.findContours` — extração de contornos geométricos
- `scipy.spatial.distance.cdist` — matriz de distâncias euclidianas
- `tkinter` + `threading` + `queue` — GUI não bloqueante
- `Pillow (PIL)` — conversão BGR→RGB para exibição no canvas Tkinter

**Nenhum modelo de IA, Deep Learning ou YOLO foi utilizado.**
