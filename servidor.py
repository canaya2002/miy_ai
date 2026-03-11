"""
Servidor web de produccion para la IA.
Maneja chat, ensenanza, auto-aprendizaje, estadisticas, y la pagina del cerebro.

Compatible con gunicorn (produccion) y Flask dev server (local).
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import logging
import uuid
import random
import sys
import torch

from cerebro import CerebroIA
import entrenar
import base_datos as bd
import auto_aprendizaje as auto
from migrar import migrar_json_a_sqlite

# ============================================================
#  CONFIGURACION DE FLASK
# ============================================================

app = Flask(__name__)
CORS(app)

# Rate limiting: maximo 60 requests por minuto por IP
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================
#  ESTADO GLOBAL DEL MODELO (thread-safe)
# ============================================================

_modelo = None
_vocabulario = []
_tags = []
_lock_modelo = threading.Lock()
_entrenando = False


def obtener_modelo():
    """Devuelve el modelo actual de forma thread-safe"""
    with _lock_modelo:
        return _modelo, list(_vocabulario), list(_tags)


def actualizar_modelo(modelo, vocabulario, tags):
    """Intercambia el modelo de forma atomica (sin interrumpir requests)"""
    global _modelo, _vocabulario, _tags
    with _lock_modelo:
        _modelo = modelo
        _vocabulario = vocabulario
        _tags = tags


def reentrenar_background():
    """
    Re-entrena el modelo en un thread de background.
    Mientras entrena, el modelo anterior sigue respondiendo.
    Cuando termina, se intercambia atomicamente.
    """
    global _entrenando

    if _entrenando:
        return

    def _entrenar():
        global _entrenando
        _entrenando = True
        try:
            logger.info("Re-entrenamiento iniciado...")
            modelo_nuevo, vocab_nuevo, tags_nuevo = entrenar.entrenar_modelo()
            actualizar_modelo(modelo_nuevo, vocab_nuevo, tags_nuevo)
            stats = bd.obtener_estadisticas()
            logger.info(
                f"Re-entrenamiento completado. "
                f"{stats['temas']} temas, {stats['patrones']} patrones."
            )
        except Exception as e:
            logger.error(f"Error en re-entrenamiento: {e}")
        finally:
            _entrenando = False

    thread = threading.Thread(target=_entrenar, daemon=True, name="reentrenamiento")
    thread.start()


# ============================================================
#  INICIALIZACION
# ============================================================

def inicializar():
    """Inicializa base de datos, migra datos, carga modelo, activa auto-aprendizaje"""
    global _modelo, _vocabulario, _tags

    logger.info("Inicializando IA...")

    # Crear tablas en SQLite
    bd.crear_tablas()
    logger.info("  Base de datos lista.")

    # Migrar datos.json a SQLite (solo la primera vez)
    migrar_json_a_sqlite()

    # Cargar modelo entrenado (o entrenar si no existe)
    try:
        _modelo, _vocabulario, _tags = entrenar.cargar_modelo()
        logger.info(f"  Modelo cargado: {len(_tags)} temas, {len(_vocabulario)} palabras.")
    except Exception as e:
        logger.error(f"  Error cargando modelo: {e}")
        logger.info("  Entrenando modelo nuevo...")
        _modelo, _vocabulario, _tags = entrenar.entrenar_modelo(verbose=True)

    # Conectar auto-aprendizaje con re-entrenamiento
    auto.registrar_callback_reentrenar(reentrenar_background)

    # Iniciar scheduler periodico (re-entrena cada 5 min si hay cambios)
    auto.iniciar_scheduler()
    logger.info("  Auto-aprendizaje activado.")

    logger.info("IA lista!")


# ============================================================
#  UTILIDADES
# ============================================================

def obtener_sesion():
    """Obtiene o crea un ID de sesion unico via cookie"""
    sesion_id = request.cookies.get("ia_sesion")
    es_nueva = False
    if not sesion_id:
        sesion_id = str(uuid.uuid4())
        es_nueva = True
    return sesion_id, es_nueva


def elegir_respuesta_ponderada(respuestas):
    """
    Elige una respuesta usando pesos (weighted random).
    Respuestas con mejor feedback aparecen mas seguido.
    """
    if not respuestas:
        return None, None
    pesos = [max(r["peso"], 0.1) for r in respuestas]
    elegida = random.choices(respuestas, weights=pesos, k=1)[0]
    return elegida["texto"], elegida["id"]


def respuesta_con_cookie(data, sesion_id, status=200):
    """Crea una respuesta JSON con la cookie de sesion"""
    resp = jsonify(data)
    resp.status_code = status
    resp.set_cookie("ia_sesion", sesion_id, max_age=86400 * 30, samesite="Lax")
    return resp


# ============================================================
#  RUTAS: PAGINAS
# ============================================================

@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/cerebro")
def pagina_cerebro():
    return render_template("cerebro.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "entrenando": _entrenando})


# ============================================================
#  API: CHAT
# ============================================================

@app.route("/chat", methods=["POST"])
@limiter.limit("30 per minute")
def chat():
    try:
        sesion_id, es_nueva = obtener_sesion()

        if es_nueva:
            bd.registrar_sesion_nueva()

        mensaje = request.json.get("mensaje", "").strip()
        if not mensaje:
            return respuesta_con_cookie({"respuesta": "", "entendio": False}, sesion_id)

        # Obtener modelo actual
        modelo, vocabulario, tags = obtener_modelo()

        if modelo is None or not tags:
            return respuesta_con_cookie({
                "respuesta": "Todavia estoy aprendiendo. Intenta en un momento.",
                "entendio": False,
                "confianza": 0
            }, sesion_id)

        # Procesar mensaje (con tolerancia a errores ortograficos)
        palabras = entrenar.limpiar(mensaje)
        bolsa = entrenar.bolsa_de_palabras(palabras, vocabulario, flexible=True)
        tensor = torch.FloatTensor(bolsa).unsqueeze(0)

        with torch.no_grad():
            resultado = modelo(tensor)

        probabilidades = torch.softmax(resultado, dim=1)
        confianza_val, prediccion = torch.max(probabilidades, dim=1)

        tag = tags[prediccion.item()]
        confianza = confianza_val.item()

        # Guardar mensaje del usuario
        bd.guardar_mensaje(sesion_id, "usuario", mensaje, tag, confianza)

        if confianza > 0.6:
            # La IA entiende: buscar respuesta ponderada por feedback
            respuestas = bd.obtener_respuestas_por_tag(tag)
            texto_respuesta, respuesta_id = elegir_respuesta_ponderada(respuestas)

            if not texto_respuesta:
                texto_respuesta = f"Se que hablas de {tag}, pero no tengo respuesta aun."
                respuesta_id = None

            # Guardar respuesta de la IA
            bd.guardar_mensaje(sesion_id, "ia", texto_respuesta, tag, confianza)

            # Auto-aprendizaje (refuerzo, feedback, contexto)
            auto.procesar_mensaje(sesion_id, mensaje, tag, confianza, respuesta_id)

            return respuesta_con_cookie({
                "respuesta": texto_respuesta,
                "entendio": True,
                "tag": tag,
                "confianza": round(confianza * 100, 1)
            }, sesion_id)
        else:
            # No entendio
            texto_respuesta = "No entendi eso. Quieres ensenarme?"
            bd.guardar_mensaje(sesion_id, "ia", texto_respuesta, None, confianza)
            auto.procesar_mensaje(sesion_id, mensaje, None, confianza)

            return respuesta_con_cookie({
                "respuesta": texto_respuesta,
                "entendio": False,
                "confianza": round(confianza * 100, 1)
            }, sesion_id)

    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        return jsonify({
            "respuesta": "Ocurrio un error interno. Intenta de nuevo.",
            "entendio": False,
            "confianza": 0
        }), 500


# ============================================================
#  API: ENSENAR
# ============================================================

@app.route("/ensenar", methods=["POST"])
@limiter.limit("10 per minute")
def ensenar_ruta():
    try:
        info = request.json
        frase = info.get("frase", "").strip()
        tag = info.get("tag", "").strip().lower()
        respuesta = info.get("respuesta", "").strip()

        if not frase or not tag:
            return jsonify({"ok": False, "error": "Faltan datos"})

        # Sanitizar tag: solo letras, numeros, guiones bajos
        tag = "".join(c for c in tag if c.isalnum() or c in "-_").strip()
        if not tag:
            return jsonify({"ok": False, "error": "Tag invalido"})

        # Auto-detectar si el tag ya existe
        conn = bd.conectar()
        existe = conn.execute("SELECT id FROM intenciones WHERE tag = ?", (tag,)).fetchone()
        conn.close()

        if existe:
            bd.agregar_patron(tag, frase, origen="manual", confianza=1.0)
            if respuesta:
                bd.agregar_respuesta(tag, respuesta)
        else:
            resp_texto = respuesta if respuesta else f"Me hablaste de {tag}!"
            bd.crear_intencion_completa(tag, [frase], [resp_texto])

        # Re-entrenar en background
        reentrenar_background()

        return jsonify({"ok": True})

    except Exception as e:
        logger.error(f"Error en /ensenar: {e}")
        return jsonify({"ok": False, "error": "Error interno"}), 500


# ============================================================
#  API: ESTADISTICAS
# ============================================================

@app.route("/stats")
def stats():
    try:
        datos = bd.obtener_estadisticas()
        datos["entrenando"] = _entrenando
        datos["vocabulario"] = len(_vocabulario)
        return jsonify(datos)
    except Exception as e:
        logger.error(f"Error en /stats: {e}")
        return jsonify({"error": "Error obteniendo estadisticas"}), 500


@app.route("/api/cerebro")
def api_cerebro():
    """API JSON con datos detallados del cerebro (para la pagina /cerebro)"""
    try:
        temas = bd.obtener_temas_detallados()
        estadisticas = bd.obtener_estadisticas()
        return jsonify({
            "temas": temas,
            "estadisticas": estadisticas
        })
    except Exception as e:
        logger.error(f"Error en /api/cerebro: {e}")
        return jsonify({"error": "Error interno"}), 500


# ============================================================
#  MANEJO DE ERRORES GLOBAL
# ============================================================

@app.errorhandler(404)
def no_encontrado(e):
    return jsonify({"error": "Ruta no encontrada"}), 404


@app.errorhandler(429)
def rate_limit_error(e):
    return jsonify({
        "error": "Demasiados mensajes. Espera un momento.",
        "respuesta": "Hey, tranquilo! Muchos mensajes muy rapido. Espera un poco.",
        "entendio": False
    }), 429


@app.errorhandler(500)
def error_interno(e):
    return jsonify({"error": "Error interno del servidor"}), 500


# ============================================================
#  INICIAR
# ============================================================

# Inicializar al importar el modulo (compatible con gunicorn: gunicorn servidor:app)
inicializar()

if __name__ == "__main__":
    logger.info("\n========================================")
    logger.info("  Servidor web listo!")
    logger.info("  Chat:    http://localhost:5000")
    logger.info("  Cerebro: http://localhost:5000/cerebro")
    logger.info("  Stats:   http://localhost:5000/stats")
    logger.info("  Health:  http://localhost:5000/health")
    logger.info("========================================\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
