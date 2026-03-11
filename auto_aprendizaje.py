"""
Auto-aprendizaje nivel maximo.
Incluye: refuerzo, contexto, feedback, clustering colectivo,
absorcion de respuestas, decay, limpieza automatica, backups.
Buffer de patrones via tabla staging en SQLite (no en memoria).
Sesiones con TTL para evitar memory leaks.
"""

import threading
import time
import re
from collections import Counter
import base_datos as bd
from entrenar import limpiar, similitud_palabras

# ============================================================
#  CONFIGURACION
# ============================================================

UMBRAL_ALTO = 0.85
UMBRAL_MEDIO = 0.60
UMBRAL_CONTEXTO = 0.50

MIN_SESIONES_CLUSTERING = 3

PALABRAS_POSITIVAS = {
    "jaja", "jajaja", "jajajaja", "si", "sí", "exacto", "buena", "gracias",
    "genial", "correcto", "eso", "bien", "perfecto", "nice", "ok", "vale",
    "cierto", "verdad", "claro", "asi es", "efectivamente", "buenisimo",
    "excelente", "crack", "god", "increible", "wow", "me gusta", "eso es",
    "a huevo", "simon", "chido", "de acuerdo", "tiene sentido"
}

PALABRAS_NEGATIVAS = {
    "no", "mal", "eso no", "equivocado", "error", "incorrecto", "nada que ver",
    "mentira", "falso", "estas mal", "no es asi", "te equivocas", "wrong",
    "nope", "nel", "para nada", "que no", "no es eso", "mal ahi",
    "te equivocaste", "no no"
}

STOP_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "y", "o", "a", "al", "es", "que", "por", "para", "con", "se",
    "me", "te", "lo", "le", "mi", "tu", "su", "nos", "como", "mas",
    "pero", "si", "no", "ya", "hay", "muy", "esta", "esto", "eso"
}

# ============================================================
#  SESIONES EN MEMORIA (con TTL y limpieza)
# ============================================================

_sesiones = {}
_sesiones_ts = {}
_lock_sesiones = threading.Lock()
MAX_SESIONES = 20000
SESION_TTL = 3600  # 1 hora


def _limpiar_sesiones_expiradas():
    """Limpia sesiones con mas de 1 hora de inactividad."""
    ahora = time.time()
    if len(_sesiones) < MAX_SESIONES // 2:
        return
    expiradas = [k for k, ts in _sesiones_ts.items() if (ahora - ts) > SESION_TTL]
    for k in expiradas:
        _sesiones.pop(k, None)
        _sesiones_ts.pop(k, None)


def obtener_estado_sesion(sesion_id):
    with _lock_sesiones:
        _limpiar_sesiones_expiradas()
        if sesion_id not in _sesiones:
            _sesiones[sesion_id] = {
                "frase_no_entendida": None,
                "ultimo_tag": None,
                "ultima_respuesta_id": None
            }
        _sesiones_ts[sesion_id] = time.time()
        return _sesiones[sesion_id].copy()


def actualizar_sesion(sesion_id, **kwargs):
    with _lock_sesiones:
        if sesion_id not in _sesiones:
            _sesiones[sesion_id] = {
                "frase_no_entendida": None,
                "ultimo_tag": None,
                "ultima_respuesta_id": None
            }
        _sesiones[sesion_id].update(kwargs)
        _sesiones_ts[sesion_id] = time.time()


# ============================================================
#  AUTO-APRENDIZAJE PRINCIPAL
# ============================================================

def procesar_mensaje(sesion_id, texto_usuario, tag_detectado, confianza, respuesta_id=None):
    estado = obtener_estado_sesion(sesion_id)
    texto_limpio = " ".join(limpiar(texto_usuario))

    # Feedback
    if estado["ultima_respuesta_id"]:
        if _es_feedback_positivo(texto_limpio):
            bd.ajustar_peso_respuesta(estado["ultima_respuesta_id"], 0.15)
            actualizar_sesion(sesion_id, ultima_respuesta_id=None)
            return "feedback_positivo"
        if _es_feedback_negativo(texto_limpio):
            bd.ajustar_peso_respuesta(estado["ultima_respuesta_id"], -0.15)
            actualizar_sesion(sesion_id, ultima_respuesta_id=None)
            return "feedback_negativo"

    # Refuerzo alto
    if confianza > UMBRAL_ALTO and tag_detectado:
        _agregar_patron_staging(tag_detectado, texto_usuario, "auto_refuerzo", confianza)
        if estado["frase_no_entendida"]:
            _agregar_patron_staging(tag_detectado, estado["frase_no_entendida"], "auto_contexto", UMBRAL_CONTEXTO)
            actualizar_sesion(sesion_id, frase_no_entendida=None)
        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "refuerzo"

    # Confianza media
    if confianza > UMBRAL_MEDIO and tag_detectado:
        if estado["frase_no_entendida"]:
            _agregar_patron_staging(tag_detectado, estado["frase_no_entendida"], "auto_contexto", UMBRAL_CONTEXTO * 0.7)
            actualizar_sesion(sesion_id, frase_no_entendida=None)
        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "respuesta_normal"

    # No entendio
    actualizar_sesion(sesion_id, frase_no_entendida=texto_usuario, ultima_respuesta_id=None)
    return "no_entendido"


def _es_feedback_positivo(t):
    return t in PALABRAS_POSITIVAS

def _es_feedback_negativo(t):
    return t in PALABRAS_NEGATIVAS


def _agregar_patron_staging(tag, texto, origen, confianza):
    """Escribe a tabla staging en SQLite en vez de memoria."""
    bd.agregar_patron_pendiente(tag, texto, origen, confianza)


# ============================================================
#  CLUSTERING DE FRASES NO ENTENDIDAS
# ============================================================

def clustering_frases_no_entendidas():
    """
    Agrupa frases no entendidas por palabras clave compartidas.
    Si 3+ usuarios diferentes preguntan cosas similares, crea un tag automatico.
    """
    frases = bd.obtener_frases_no_entendidas(dias=7)
    if len(frases) < MIN_SESIONES_CLUSTERING:
        return 0

    grupos = {}
    for f in frases:
        palabras = set(limpiar(f["texto"])) - STOP_WORDS
        palabras = {p for p in palabras if len(p) > 2}
        for palabra in palabras:
            if palabra not in grupos:
                grupos[palabra] = {"frases": [], "sesiones": set()}
            grupos[palabra]["frases"].append(f["texto"])
            grupos[palabra]["sesiones"].add(f["sesion"])

    creados = 0
    tags_existentes = set(bd.obtener_estadisticas()["lista_temas"])

    for palabra, info in grupos.items():
        if len(info["sesiones"]) >= MIN_SESIONES_CLUSTERING and palabra not in tags_existentes:
            frases_unicas = list(set(info["frases"]))[:10]
            respuesta = f"Me han preguntado mucho sobre {palabra} pero todavia no se que responder. Me ensenas?"
            resultado = bd.crear_intencion_completa(palabra, frases_unicas, [respuesta])
            if resultado is not None:
                creados += 1
                tags_existentes.add(palabra)
                print(f"  [clustering] Tag auto-creado: '{palabra}' ({len(frases_unicas)} patrones)")

    return creados


# ============================================================
#  ABSORCION DE RESPUESTAS
# ============================================================

def absorber_respuesta(tag, texto_respuesta):
    """
    Extrae palabras clave de una respuesta ensenada y las agrega
    como patrones debiles del mismo tag.
    """
    palabras = set(limpiar(texto_respuesta)) - STOP_WORDS
    palabras = {p for p in palabras if len(p) > 3}

    agregados = 0
    for palabra in palabras:
        if bd.agregar_patron(tag, palabra, origen="auto_absorcion", confianza=0.3):
            agregados += 1

    return agregados


# ============================================================
#  MANTENIMIENTO DIARIO
# ============================================================

def ejecutar_mantenimiento():
    """
    Corre una vez al dia:
    1. Decay de confianza en patrones sin uso
    2. Limpieza de patrones basura
    3. Desactivar respuestas malas
    4. Archivar intenciones vacias
    5. Clustering de frases no entendidas
    6. Purga de mensajes viejos
    7. Backup de la base de datos
    """
    print("[mantenimiento] Iniciando mantenimiento diario...")

    afectados = bd.decay_confianza_inactivos(dias=30, factor=0.95)
    if afectados:
        print(f"  [decay] {afectados} patrones con confianza reducida")

    eliminados = bd.limpiar_patrones_basura(min_confianza=0.1, dias=60)
    if eliminados:
        print(f"  [limpieza] {eliminados} patrones basura eliminados")

    desactivadas = bd.desactivar_respuestas_malas(min_peso=0.2)
    if desactivadas:
        print(f"  [limpieza] {desactivadas} respuestas desactivadas")

    vacias = bd.archivar_intenciones_vacias()
    if vacias:
        print(f"  [limpieza] {vacias} intenciones vacias archivadas")

    nuevos_tags = clustering_frases_no_entendidas()
    if nuevos_tags:
        print(f"  [clustering] {nuevos_tags} tags nuevos auto-creados")

    eliminados_msg = bd.purgar_mensajes_viejos(dias=30)
    if eliminados_msg:
        print(f"  [purga] {eliminados_msg} mensajes viejos eliminados")

    bd.backup_db()
    print("[mantenimiento] Completado.")

    return {"decay": afectados, "eliminados": eliminados, "desactivadas": desactivadas,
            "vacias": vacias, "nuevos_tags": nuevos_tags, "mensajes_purgados": eliminados_msg}
