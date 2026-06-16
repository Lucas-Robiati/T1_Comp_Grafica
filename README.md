# T1 Computação Gráfica — Video Object Mapper

Sistema completo em Python 3 com interface gráfica para detecção, rastreamento e análise de objetos em vídeos, utilizando YOLOv8 e técnicas de computação gráfica.

## Funcionalidades

- **Detecção de objetos** com YOLOv8 (dataset COCO — 80 classes)
- **Rastreamento** de objetos únicos ao longo do vídeo (tracker IoU)
- **Análise de vibes** — classifica cada objeto por cor predominante e velocidade
- **Notificação SMS** via Twilio ao término do processamento (opcional)
- **Interface gráfica** com barra de progresso, log em tempo real e cancelamento
- **Otimizado** para vídeos longos (1-2h) com frame skipping configurável

## Requisitos do Sistema

- Python 3.8+
- `tkinter` (geralmente incluído no Python; no Linux: `sudo apt-get install python3-tk`)
- Conexão com internet na primeira execução (download automático do modelo YOLOv8)

## Instalação

```bash
# 1. Clonar o repositório
git clone <url-do-repositório>
cd T1_Comp_Grafica

# 2. Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

A interface gráfica será aberta. Selecione um arquivo de vídeo (.mp4, .avi, .mov) e clique em "Processar Vídeo".

## Configuração

Edite `config.yaml` para ajustar parâmetros:

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `frame_interval_seconds` | Intervalo entre frames processados | `2` |
| `confidence_threshold` | Confiança mínima do YOLO | `0.4` |
| `model_name` | Modelo YOLOv8 | `yolov8n.pt` |
| `iou_threshold` | Threshold IoU para o tracker | `0.3` |
| `max_age` | Frames sem detecção antes de perder track | `10` |

### Twilio (Opcional)

Para receber SMS ao término do processamento, configure as credenciais no `config.yaml` ou via variáveis de ambiente:

```yaml
twilio:
  account_sid: "ACxxxx..."
  auth_token: "xxxx..."
  from_phone: "+5511999999999"
  to_phone: "+5511888888888"
```

Ou via `.env`:
```
TWILIO_ACCOUNT_SID=ACxxxx...
TWILIO_AUTH_TOKEN=xxxx...
TWILIO_FROM_PHONE=+5511999999999
TWILIO_TO_PHONE=+5511888888888
```

## Arquivos de Saída

Gerados no diretório `output/`:

| Arquivo | Descrição |
|---------|-----------|
| `object_map.json` | Mapeamento completo de todos os objetos com IDs, classes, timestamps e bboxes |
| `vibe_analysis.csv` | Tabela com cada objeto, classe, vibe e velocidade média |
| `vibe_prospecting_report.txt` | Relatório com estatísticas de vibes e insights de marketing |

## Estrutura do Projeto

```
├── main.py                  # Ponto de entrada
├── config.yaml              # Configurações
├── requirements.txt         # Dependências
├── gui/
│   └── app.py               # Interface Tkinter
├── engine/
│   ├── detector.py           # Wrapper YOLOv8
│   ├── tracker.py            # Tracker IoU
│   └── video_processor.py    # Orquestrador do pipeline
├── plugins/
│   ├── twilio_notifier.py    # Notificação SMS
│   └── vibe_prospecting.py   # Análise de vibes
├── utils/
│   ├── logger.py             # Setup de logging
│   └── config_loader.py      # Carregamento de config
└── tests/
    ├── test_tracker.py       # Testes do tracker
    └── test_vibe.py          # Testes de vibe
```

## Testes

```bash
python -m pytest tests/ -v
```