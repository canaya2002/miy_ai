"""
Servidor de produccion: seguridad, SEO, cache, monitoreo, adaptive throttling.
Optimizado para miles de usuarios con gevent workers.
Compresion GZIP delegada a Nginx. Entrenamiento delegado a entrenador_worker.
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import logging
import uuid
import random
import re
import os
import sys
import time
import json
import hashlib
import torch
from collections import OrderedDict

from cerebro import CerebroIA
import entrenar
import base_datos as bd
import auto_aprendizaje as auto
from migrar import migrar_json_a_sqlite

# ============================================================
#  APP
# ============================================================

app = Flask(__name__)

CORS(app, origins=["https://*.onrender.com", "http://localhost:5000", "http://127.0.0.1:5000"])

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"], storage_uri="memory://")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

_inicio_servidor = time.time()

# ============================================================
#  SIGNAL FILES
# ============================================================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_RETRAIN = os.path.join(_BASE_DIR, "NEEDS_RETRAIN")
SIGNAL_UPDATED = os.path.join(_BASE_DIR, "MODEL_UPDATED")


def solicitar_reentrenamiento():
    try:
        with open(SIGNAL_RETRAIN, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def verificar_modelo_actualizado():
    if os.path.exists(SIGNAL_UPDATED):
        try:
            os.remove(SIGNAL_UPDATED)
            modelo_nuevo, vocab_nuevo, tags_nuevo = entrenar.cargar_modelo()
            actualizar_modelo(modelo_nuevo, vocab_nuevo, tags_nuevo)
            logger.info("Modelo recargado en caliente.")
        except Exception as e:
            logger.error(f"Error recargando modelo: {e}")


# ============================================================
#  ADAPTIVE THROTTLING
# ============================================================

_sesiones_activas = {}
_sesiones_activas_lock = threading.Lock()
VENTANA_ACTIVIDAD = 300  # 5 minutos


def _registrar_actividad(sesion_id):
    with _sesiones_activas_lock:
        _sesiones_activas[sesion_id] = time.time()


def _contar_usuarios_activos():
    ahora = time.time()
    with _sesiones_activas_lock:
        # Limpiar viejos
        expirados = [k for k, v in _sesiones_activas.items() if ahora - v > VENTANA_ACTIVIDAD]
        for k in expirados:
            del _sesiones_activas[k]
        return len(_sesiones_activas)


def _obtener_limite_chat():
    activos = _contar_usuarios_activos()
    if activos < 20:
        return "60 per minute"
    elif activos < 100:
        return "30 per minute"
    return "15 per minute"


# Adaptive cache TTLs
def _stats_ttl():
    activos = _contar_usuarios_activos()
    return 60 if activos > 50 else 30


def _cerebro_ttl():
    activos = _contar_usuarios_activos()
    return 120 if activos > 50 else 60


# ============================================================
#  SEGURIDAD: SANITIZACION
# ============================================================

def sanitizar(texto, max_largo=500):
    if not isinstance(texto, str):
        return ""
    texto = texto[:max_largo]
    texto = ''.join(c for c in texto if c.isprintable() or c == ' ')
    texto = texto.replace('<', '&lt;').replace('>', '&gt;')
    texto = ' '.join(texto.split())
    return texto.strip()


def contenido_permitido(texto):
    if 'http://' in texto or 'https://' in texto or 'www.' in texto:
        return False
    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', texto):
        return False
    if re.search(r'\d{10,}', texto.replace(' ', '').replace('-', '')):
        return False
    return True


# ============================================================
#  SEGURIDAD: HEADERS
# ============================================================

@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    csp = ("default-src 'self'; "
           "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
           "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://api.fontshare.com; "
           "font-src https://fonts.gstatic.com https://cdn.fontshare.com; "
           "connect-src 'self'; "
           "img-src 'self' data:; ")
    response.headers['Content-Security-Policy'] = csp
    return response


# ============================================================
#  CACHE
# ============================================================

_stats_cache = {"data": None, "ts": 0, "etag": ""}
_cerebro_cache = {"data": None, "ts": 0, "etag": ""}

# ============================================================
#  CACHE DE CLASIFICACION (5000 entradas)
# ============================================================

_cache = OrderedDict()
_cache_max = 5000
_lock_cache = threading.Lock()


def clasificar_con_cache(texto_normalizado, modelo, vocabulario, tags):
    with _lock_cache:
        if texto_normalizado in _cache:
            _cache.move_to_end(texto_normalizado)
            return _cache[texto_normalizado]

    palabras = entrenar.limpiar(texto_normalizado)
    # Aplicar sinonimos
    mapa_sin = bd.obtener_sinonimos()
    palabras = entrenar.aplicar_sinonimos(palabras, mapa_sin)

    bolsa = entrenar.bolsa_de_palabras(palabras, vocabulario, flexible=True)
    tensor = torch.FloatTensor(bolsa).unsqueeze(0)

    with torch.no_grad():
        resultado = modelo(tensor)
    probs = torch.softmax(resultado, dim=1)
    conf_val, pred = torch.max(probs, dim=1)
    tag = tags[pred.item()]
    confianza = conf_val.item()

    with _lock_cache:
        _cache[texto_normalizado] = (tag, confianza)
        if len(_cache) > _cache_max:
            _cache.popitem(last=False)

    return tag, confianza


def invalidar_cache(tag=None):
    """Invalida todo el cache, o solo entradas de un tag."""
    with _lock_cache:
        if tag is None:
            _cache.clear()
        else:
            to_remove = [k for k, v in _cache.items() if v[0] == tag]
            for k in to_remove:
                del _cache[k]


# ============================================================
#  MODELO
# ============================================================

_modelo = None
_vocabulario = []
_tags = []
_lock_modelo = threading.Lock()


def obtener_modelo():
    with _lock_modelo:
        return _modelo, list(_vocabulario), list(_tags)


def actualizar_modelo(modelo, vocabulario, tags):
    global _modelo, _vocabulario, _tags
    with _lock_modelo:
        _modelo = modelo
        _vocabulario = vocabulario
        _tags = tags
    invalidar_cache()


# ============================================================
#  INICIALIZACION
# ============================================================

def inicializar():
    global _modelo, _vocabulario, _tags
    logger.info("Inicializando IA...")
    bd.crear_tablas()
    migrar_json_a_sqlite()
    try:
        _modelo, _vocabulario, _tags = entrenar.cargar_modelo()
        logger.info(f"  Modelo: {len(_tags)} temas, {len(_vocabulario)} palabras.")
    except Exception as e:
        logger.error(f"  Error modelo: {e}")
        _modelo, _vocabulario, _tags = entrenar.entrenar_modelo(verbose=True)
    logger.info("IA lista!")


# ============================================================
#  BEFORE REQUEST
# ============================================================

_last_model_check = 0
MODEL_CHECK_INTERVAL = 10


@app.before_request
def antes_de_request():
    global _last_model_check
    bd.flush_mensajes_periodico()
    ahora = time.time()
    if ahora - _last_model_check > MODEL_CHECK_INTERVAL:
        _last_model_check = ahora
        verificar_modelo_actualizado()


# ============================================================
#  UTILIDADES
# ============================================================

def obtener_sesion():
    sesion_id = request.cookies.get("ia_sesion")
    es_nueva = False
    if not sesion_id:
        sesion_id = str(uuid.uuid4())
        es_nueva = True
    return sesion_id, es_nueva


def elegir_respuesta_ponderada(respuestas):
    if not respuestas:
        return None, None
    pesos = [max(r["peso"], 0.1) for r in respuestas]
    elegida = random.choices(respuestas, weights=pesos, k=1)[0]
    return elegida["texto"], elegida["id"]


def respuesta_con_cookie(data, sesion_id, status=200):
    resp = jsonify(data)
    resp.status_code = status
    resp.set_cookie("ia_sesion", sesion_id, max_age=86400*30, samesite="Lax", httponly=True)
    return resp


def _generar_etag(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


# ============================================================
#  PAGINAS
# ============================================================

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/cerebro")
def pagina_cerebro():
    return render_template("cerebro.html")


# ============================================================
#  SEO
# ============================================================

@app.route("/robots.txt")
def robots():
    host = request.host_url.rstrip('/')
    txt = f"User-agent: *\nAllow: /\nAllow: /cerebro\nDisallow: /api/\nDisallow: /health\nSitemap: {host}/sitemap.xml\n"
    return Response(txt, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    host = request.host_url.rstrip('/')
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{host}/</loc><changefreq>daily</changefreq><priority>1.0</priority></url>
  <url><loc>{host}/cerebro</loc><changefreq>daily</changefreq><priority>0.8</priority></url>
</urlset>"""
    return Response(xml, mimetype="application/xml")


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "IA Experimental",
        "short_name": "IA",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#06060c",
        "theme_color": "#06060c",
        "description": "IA conversacional que aprende de cada persona"
    })


# ============================================================
#  HEALTH
# ============================================================

@app.route("/health")
def health():
    info = bd.obtener_info_salud()
    uptime = int(time.time() - _inicio_servidor)
    return jsonify({
        "status": "ok",
        "uptime_seconds": uptime,
        "modelo_cargado": _modelo is not None,
        "entrenando": os.path.exists(SIGNAL_RETRAIN),
        "usuarios_activos": _contar_usuarios_activos(),
        **info
    })


# ============================================================
#  API: CHAT
# ============================================================

@app.route("/chat", methods=["POST"])
@limiter.limit(lambda: _obtener_limite_chat())
def chat():
    try:
        sesion_id, es_nueva = obtener_sesion()
        if es_nueva:
            bd.registrar_sesion_nueva()

        _registrar_actividad(sesion_id)

        raw = request.json.get("mensaje", "") if request.json else ""
        mensaje = sanitizar(raw, max_largo=500)
        if not mensaje:
            return respuesta_con_cookie({"respuesta": "", "entendio": False}, sesion_id)

        modelo, vocabulario, tags = obtener_modelo()
        if modelo is None or not tags:
            return respuesta_con_cookie({
                "respuesta": "Todavia estoy aprendiendo. Intenta en un momento.",
                "entendio": False, "confianza": 0
            }, sesion_id)

        texto_norm = mensaje.lower().strip()
        tag, confianza = clasificar_con_cache(texto_norm, modelo, vocabulario, tags)

        bd.guardar_mensaje(sesion_id, "usuario", mensaje, tag, confianza)

        if confianza > 0.6:
            respuestas = bd.obtener_respuestas_por_tag(tag)
            texto_resp, resp_id = elegir_respuesta_ponderada(respuestas)
            if not texto_resp:
                texto_resp = f"Se que hablas de {tag}, pero no tengo respuesta aun."
                resp_id = None
            bd.guardar_mensaje(sesion_id, "ia", texto_resp, tag, confianza)
            auto.procesar_mensaje(sesion_id, mensaje, tag, confianza, resp_id)
            return respuesta_con_cookie({
                "respuesta": texto_resp, "entendio": True,
                "tag": tag, "confianza": round(confianza * 100, 1)
            }, sesion_id)
        else:
            texto_resp = "No entendi eso. Quieres ensenarme?"
            bd.guardar_mensaje(sesion_id, "ia", texto_resp, None, confianza)
            auto.procesar_mensaje(sesion_id, mensaje, tag if confianza > 0.3 else None, confianza)
            return respuesta_con_cookie({
                "respuesta": texto_resp, "entendio": False,
                "confianza": round(confianza * 100, 1)
            }, sesion_id)

    except Exception as e:
        logger.error(f"Error /chat: {e}", exc_info=True)
        return jsonify({"respuesta": "Algo salio mal. Intenta de nuevo.", "entendio": False, "confianza": 0}), 500


# ============================================================
#  API: ENSENAR
# ============================================================

@app.route("/ensenar", methods=["POST"])
@limiter.limit("5 per minute")
def ensenar_ruta():
    try:
        info = request.json if request.json else {}
        frase = sanitizar(info.get("frase", ""), max_largo=200)
        tag = sanitizar(info.get("tag", ""), max_largo=50)
        respuesta = sanitizar(info.get("respuesta", ""), max_largo=500)

        tag = "".join(c for c in tag.lower() if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        if not frase or not tag:
            return jsonify({"ok": False, "error": "Faltan datos"})
        if len(tag) < 2:
            return jsonify({"ok": False, "error": "El tema debe tener al menos 2 caracteres"})
        if len(frase) < 2:
            return jsonify({"ok": False, "error": "La frase es muy corta"})

        if not contenido_permitido(frase):
            return jsonify({"ok": False, "error": "Contenido no permitido en la frase"})
        if respuesta and not contenido_permitido(respuesta):
            return jsonify({"ok": False, "error": "Contenido no permitido en la respuesta"})

        limites = bd.verificar_limites_ensenanza(tag)
        if limites["intenciones_lleno"]:
            return jsonify({"ok": False, "error": "La IA ya sabe demasiados temas. No puede aprender mas por ahora."})
        if limites["patrones_lleno"]:
            return jsonify({"ok": False, "error": "Este tema ya tiene muchos patrones."})

        conn = bd.conectar()
        existe = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
        conn.close()

        if existe:
            bd.agregar_patron(tag, frase, origen="manual", confianza=1.0)
            if respuesta:
                bd.agregar_respuesta(tag, respuesta)
                auto.absorber_respuesta(tag, respuesta)
        else:
            resp_texto = respuesta if respuesta else f"Me hablaste de {tag}!"
            bd.crear_intencion_completa(tag, [frase], [resp_texto])
            if respuesta:
                auto.absorber_respuesta(tag, respuesta)

        invalidar_cache(tag)
        solicitar_reentrenamiento()
        return jsonify({"ok": True})

    except Exception as e:
        logger.error(f"Error /ensenar: {e}", exc_info=True)
        return jsonify({"ok": False, "error": "Error interno"}), 500


# ============================================================
#  API: STATS (con ETag)
# ============================================================

@app.route("/stats")
def stats():
    try:
        ahora = time.time()
        ttl = _stats_ttl()
        if _stats_cache["data"] and (ahora - _stats_cache["ts"]) < ttl:
            datos = _stats_cache["data"].copy()
        else:
            datos = bd.obtener_estadisticas()
            datos["vocabulario"] = len(_vocabulario)
            etag = _generar_etag(datos)
            _stats_cache["data"] = datos
            _stats_cache["ts"] = ahora
            _stats_cache["etag"] = etag

        datos["entrenando"] = os.path.exists(SIGNAL_RETRAIN)

        # ETag check
        if_none = request.headers.get("If-None-Match")
        if if_none and if_none == _stats_cache["etag"]:
            return Response(status=304)

        resp = jsonify(datos)
        resp.headers["ETag"] = _stats_cache["etag"]
        return resp
    except Exception as e:
        logger.error(f"Error /stats: {e}")
        return jsonify({"error": "Error"}), 500


# ============================================================
#  API: CEREBRO (con ETag)
# ============================================================

@app.route("/api/cerebro")
def api_cerebro():
    try:
        ahora = time.time()
        ttl = _cerebro_ttl()
        if _cerebro_cache["data"] and (ahora - _cerebro_cache["ts"]) < ttl:
            if_none = request.headers.get("If-None-Match")
            if if_none and if_none == _cerebro_cache["etag"]:
                return Response(status=304)
            resp = jsonify(_cerebro_cache["data"])
            resp.headers["ETag"] = _cerebro_cache["etag"]
            return resp

        temas = bd.obtener_temas_detallados()
        estadisticas = bd.obtener_estadisticas()

        palabras_por_tag = bd.obtener_palabras_por_tag()
        tags_list = list(palabras_por_tag.keys())
        conexiones = []
        for i in range(len(tags_list)):
            for j in range(i + 1, len(tags_list)):
                shared = palabras_por_tag[tags_list[i]] & palabras_por_tag[tags_list[j]]
                if len(shared) >= 1:
                    conexiones.append({"from": tags_list[i], "to": tags_list[j], "peso": len(shared)})

        resultado = {"temas": temas, "estadisticas": estadisticas, "conexiones": conexiones}
        etag = _generar_etag(resultado)
        _cerebro_cache["data"] = resultado
        _cerebro_cache["ts"] = ahora
        _cerebro_cache["etag"] = etag

        resp = jsonify(resultado)
        resp.headers["ETag"] = etag
        return resp
    except Exception as e:
        logger.error(f"Error /api/cerebro: {e}")
        return jsonify({"error": "Error interno"}), 500


# ============================================================
#  API: CAMBIOS DESDE TIMESTAMP (para bienvenida)
# ============================================================

@app.route("/api/cambios")
def api_cambios():
    try:
        desde = request.args.get("desde", "")
        if not desde:
            return jsonify({"error": "Parametro 'desde' requerido"}), 400
        resultado = bd.obtener_cambios_desde(desde)
        return jsonify(resultado)
    except Exception as e:
        logger.error(f"Error /api/cambios: {e}")
        return jsonify({"error": "Error interno"}), 500


# ============================================================
#  SSE: STREAM DE STATS
# ============================================================

@app.route("/stream/stats")
def stream_stats():
    def generar():
        while True:
            try:
                ahora = time.time()
                ttl = _stats_ttl()
                if _stats_cache["data"] and (ahora - _stats_cache["ts"]) < ttl:
                    datos = _stats_cache["data"]
                else:
                    datos = bd.obtener_estadisticas()
                    datos["vocabulario"] = len(_vocabulario)
                    _stats_cache["data"] = datos
                    _stats_cache["ts"] = ahora
                datos_enviar = dict(datos)
                datos_enviar["entrenando"] = os.path.exists(SIGNAL_RETRAIN)
                yield f"data: {json.dumps(datos_enviar)}\n\n"
            except Exception:
                yield f"data: {{}}\n\n"
            time.sleep(30)

    return Response(generar(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================================
#  ERRORES
# ============================================================

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"404: {request.path} desde {request.remote_addr}")
    return jsonify({"error": "Ruta no encontrada"}), 404

@app.errorhandler(429)
def rate_limited(e):
    logger.warning(f"Rate limit: {request.path} desde {request.remote_addr}")
    return jsonify({"error": "Demasiadas solicitudes. Espera un momento.",
                    "respuesta": "Muchos mensajes muy rapido. Espera un poco.", "entendio": False}), 429

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Error interno del servidor"}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Error no manejado: {e}", exc_info=True)
    return jsonify({"error": "Algo salio mal."}), 500


# ============================================================
#  INICIAR
# ============================================================

inicializar()

if __name__ == "__main__":
    logger.info("\n========================================")
    logger.info("  http://localhost:5000")
    logger.info("========================================\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
