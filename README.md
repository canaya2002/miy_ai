# IA Experimental

Una inteligencia artificial conversacional construida **desde cero** con PyTorch.
No usa GPT, BERT, ni modelos pre-entrenados. Todo lo que sabe se lo enseno la gente.

## Correr local

```bash
# Instalar dependencias (torch ya debe estar instalado)
pip install -r requirements.txt

# Primera vez: entrenar el modelo con datos.json
python entrenar.py

# Iniciar servidor web
python servidor.py
```

Abrir http://localhost:5000

## Endpoints

| Ruta | Metodo | Descripcion |
|------|--------|-------------|
| `/` | GET | Chat principal |
| `/cerebro` | GET | Lo que sabe la IA (publico) |
| `/chat` | POST | Enviar mensaje |
| `/ensenar` | POST | Ensenar algo nuevo |
| `/stats` | GET | Estadisticas en vivo |
| `/health` | GET | Health check |
| `/api/cerebro` | GET | API JSON del cerebro |

## Deploy en Render

1. Sube el proyecto a un repo de GitHub
2. En [render.com](https://render.com), crea un nuevo **Web Service**
3. Conecta tu repo de GitHub
4. Render detecta `render.yaml` automaticamente
5. El disco persistente `/data/` guarda la base de datos y el modelo entrenado

## Stack

- Python 3.11
- PyTorch (CPU)
- Flask + Gunicorn
- SQLite (persistente)
- HTML/CSS/JS vanilla
