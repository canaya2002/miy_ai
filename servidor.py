"""
Servidor de produccion: seguridad blindada, SEO, cache, compresion, monitoreo.
"""

from flask import Flask, render_template, request, jsonify, make_response, Response
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
import gzip
from io import BytesIO
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

# CORS: en produccion solo el dominio propio
CORS(app, origins=["https://*.onrender.com", "http://localhost:5000", "http://127.0.0.1:5000"])

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"], storage_uri="memory://")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

_inicio_servidor = time.time()

# ============================================================
#  SEGURIDAD: SANITIZACION
# ============================================================

def sanitizar(texto, max_largo=500):
    """Limpia y valida todo input del usuario"""
    if not isinstance(texto, str):
        return ""
    texto = texto[:max_largo]
    texto = ''.join(c for c in texto if c.isprintable() or c == ' ')
    texto = texto.replace('<', '&lt;').replace('>', '&gt;')
    texto = ' '.join(texto.split())
    return texto.strip()


def contenido_permitido(texto):
    """Verifica que el contenido es aceptable para aprendizaje"""
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
           "script-src 'self' 'unsafe-inline'; "
           "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://api.fontshare.com; "
           "font-src https://fonts.gstatic.com https://cdn.fontshare.com; "
           "connect-src 'self'; "
           "img-src 'self' data:; ")
    response.headers['Content-Security-Policy'] = csp
    return response


# ============================================================
#  COMPRESION GZIP
# ============================================================

@app.after_request
def comprimir(response):
    if response.status_code < 200 or response.status_code >= 300:
        return response
    if response.direct_passthrough:
        return response
    if 'gzip' not in request.headers.get('Accept-Encoding', ''):
        return response
    content = response.get_data()
    if len(content) < 500:
        return response
    buf = BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
        f.write(content)
    response.set_data(buf.getvalue())
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(response.get_data())
    response.headers['Vary'] = 'Accept-Encoding'
    return response


# ============================================================
#  CACHE DE CLASIFICACION
# ============================================================

_cache = OrderedDict()
_cache_max = 1000
_lock_cache = threading.Lock()


def clasificar_con_cache(texto_normalizado, modelo, vocabulario, tags):
    """Cachea clasificaciones para no recalcular lo mismo"""
    with _lock_cache:
        if texto_normalizado in _cache:
            _cache.move_to_end(texto_normalizado)
            return _cache[texto_normalizado]

    palabras = entrenar.limpiar(texto_normalizado)
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


def invalidar_cache():
    with _lock_cache:
        _cache.clear()


# ============================================================
#  MODELO
# ============================================================

_modelo = None
_vocabulario = []
_tags = []
_lock_modelo = threading.Lock()
_entrenando = False


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


def reentrenar_background():
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
            logger.info(f"Re-entrenamiento OK. {stats['temas']} temas, {stats['patrones']} patrones.")
        except Exception as e:
            logger.error(f"Error re-entrenamiento: {e}")
        finally:
            _entrenando = False

    threading.Thread(target=_entrenar, daemon=True, name="reentrenamiento").start()


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
    auto.registrar_callback_reentrenar(reentrenar_background)
    auto.iniciar_scheduler()
    logger.info("IA lista!")


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
    resp.set_cookie("ia_sesion", sesion_id, max_age=86400*30, samesite="Lax")
    return resp


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
#  SEO: SITEMAP, ROBOTS, MANIFEST
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
        "background_color": "#050508",
        "theme_color": "#050508",
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
        "entrenando": _entrenando,
        "uptime_seconds": uptime,
        "modelo_cargado": _modelo is not None,
        **info
    })


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

        # Clasificar (con cache)
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
            auto.procesar_mensaje(sesion_id, mensaje, None, confianza)
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
@limiter.limit("10 per minute")
def ensenar_ruta():
    try:
        info = request.json if request.json else {}
        frase = sanitizar(info.get("frase", ""), max_largo=200)
        tag = sanitizar(info.get("tag", ""), max_largo=50)
        respuesta = sanitizar(info.get("respuesta", ""), max_largo=500)

        # Validaciones de seguridad
        tag = "".join(c for c in tag.lower() if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        if not frase or not tag:
            return jsonify({"ok": False, "error": "Faltan datos"})
        if len(tag) < 2:
            return jsonify({"ok": False, "error": "El tema debe tener al menos 2 caracteres"})
        if len(frase) < 2:
            return jsonify({"ok": False, "error": "La frase es muy corta"})

        # Filtro de contenido
        if not contenido_permitido(frase):
            return jsonify({"ok": False, "error": "Contenido no permitido en la frase"})
        if respuesta and not contenido_permitido(respuesta):
            return jsonify({"ok": False, "error": "Contenido no permitido en la respuesta"})

        # Verificar limites
        limites = bd.verificar_limites_ensenanza(tag)
        if limites["intenciones_lleno"]:
            return jsonify({"ok": False, "error": "La IA ya sabe demasiados temas. No puede aprender mas por ahora."})
        if limites["patrones_lleno"]:
            return jsonify({"ok": False, "error": "Este tema ya tiene muchos patrones."})

        # Auto-detectar si existe
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

        reentrenar_background()
        return jsonify({"ok": True})

    except Exception as e:
        logger.error(f"Error /ensenar: {e}", exc_info=True)
        return jsonify({"ok": False, "error": "Error interno"}), 500


# ============================================================
#  API: STATS Y CEREBRO
# ============================================================

@app.route("/stats")
def stats():
    try:
        datos = bd.obtener_estadisticas()
        datos["entrenando"] = _entrenando
        datos["vocabulario"] = len(_vocabulario)
        return jsonify(datos)
    except Exception as e:
        logger.error(f"Error /stats: {e}")
        return jsonify({"error": "Error"}), 500


@app.route("/api/cerebro")
def api_cerebro():
    try:
        temas = bd.obtener_temas_detallados()
        estadisticas = bd.obtener_estadisticas()

        # Calcular conexiones entre temas (palabras compartidas)
        palabras_por_tag = bd.obtener_palabras_por_tag()
        tags_list = list(palabras_por_tag.keys())
        conexiones = []
        for i in range(len(tags_list)):
            for j in range(i + 1, len(tags_list)):
                shared = palabras_por_tag[tags_list[i]] & palabras_por_tag[tags_list[j]]
                if len(shared) >= 1:
                    conexiones.append({"from": tags_list[i], "to": tags_list[j], "peso": len(shared)})

        return jsonify({"temas": temas, "estadisticas": estadisticas, "conexiones": conexiones})
    except Exception as e:
        logger.error(f"Error /api/cerebro: {e}")
        return jsonify({"error": "Error interno"}), 500


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
