"""
Auto-aprendizaje nivel maximo.
Incluye: refuerzo, contexto, feedback, clustering colectivo,
absorcion de respuestas, decay, limpieza automatica, backups.
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

INTERVALO_REENTRENAMIENTO = 300     # 5 minutos
INTERVALO_MANTENIMIENTO = 86400     # 24 horas
MAX_BUFFER_PATRONES = 20
MIN_SESIONES_CLUSTERING = 3         # minimo 3 personas diferentes para auto-crear tag

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

# Palabras comunes que se ignoran en clustering
STOP_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "en", "y", "o", "a", "al", "es", "que", "por", "para", "con", "se",
    "me", "te", "lo", "le", "mi", "tu", "su", "nos", "como", "mas",
    "pero", "si", "no", "ya", "hay", "muy", "esta", "esto", "eso"
}

# ============================================================
#  ESTADO EN MEMORIA
# ============================================================

_buffer_patrones = []
_lock_buffer = threading.Lock()
_sesiones = {}
_lock_sesiones = threading.Lock()
_callback_reentrenar = None
_entrenamiento_pendiente = False
_ultimo_mantenimiento = 0


def registrar_callback_reentrenar(callback):
    global _callback_reentrenar
    _callback_reentrenar = callback


# ============================================================
#  SESIONES
# ============================================================

def obtener_estado_sesion(sesion_id):
    with _lock_sesiones:
        if sesion_id not in _sesiones:
            _sesiones[sesion_id] = {
                "frase_no_entendida": None,
                "ultimo_tag": None,
                "ultima_respuesta_id": None
            }
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
        _agregar_patron_buffer(tag_detectado, texto_usuario, "auto_refuerzo", confianza)
        if estado["frase_no_entendida"]:
            _agregar_patron_buffer(tag_detectado, estado["frase_no_entendida"], "auto_contexto", UMBRAL_CONTEXTO)
            actualizar_sesion(sesion_id, frase_no_entendida=None)
        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "refuerzo"

    # Confianza media
    if confianza > UMBRAL_MEDIO and tag_detectado:
        if estado["frase_no_entendida"]:
            _agregar_patron_buffer(tag_detectado, estado["frase_no_entendida"], "auto_contexto", UMBRAL_CONTEXTO * 0.7)
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


def _agregar_patron_buffer(tag, texto, origen, confianza):
    global _entrenamiento_pendiente
    with _lock_buffer:
        _buffer_patrones.append({"tag": tag, "texto": texto, "origen": origen, "confianza": confianza})
        if len(_buffer_patrones) >= MAX_BUFFER_PATRONES:
            _entrenamiento_pendiente = True


# ============================================================
#  1.1 — CLUSTERING DE FRASES NO ENTENDIDAS
# ============================================================

def clustering_frases_no_entendidas():
    """
    Agrupa frases no entendidas por palabras clave compartidas.
    Si 3+ usuarios diferentes preguntan cosas similares, crea un tag automatico.
    """
    frases = bd.obtener_frases_no_entendidas(dias=7)
    if len(frases) < MIN_SESIONES_CLUSTERING:
        return 0

    # Extraer palabras clave por frase (sin stop words)
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
            # Crear tag con las frases como patrones
            frases_unicas = list(set(info["frases"]))[:10]  # max 10 patrones iniciales
            respuesta = f"Me han preguntado mucho sobre {palabra} pero todavia no se que responder. Me ensenas?"
            resultado = bd.crear_intencion_completa(palabra, frases_unicas, [respuesta])
            if resultado is not None:
                creados += 1
                tags_existentes.add(palabra)
                print(f"  [clustering] Tag auto-creado: '{palabra}' ({len(frases_unicas)} patrones)")

    return creados


# ============================================================
#  1.2 — ABSORCION DE RESPUESTAS
# ============================================================

def absorber_respuesta(tag, texto_respuesta):
    """
    Extrae palabras clave de una respuesta ensenada y las agrega
    como patrones debiles del mismo tag. Asi la IA entiende mejor.
    """
    palabras = set(limpiar(texto_respuesta)) - STOP_WORDS
    palabras = {p for p in palabras if len(p) > 3}

    agregados = 0
    for palabra in palabras:
        if bd.agregar_patron(tag, palabra, origen="auto_absorcion", confianza=0.3):
            agregados += 1

    return agregados


# ============================================================
#  1.3 + 1.4 — MANTENIMIENTO DIARIO
# ============================================================

def ejecutar_mantenimiento():
    """
    Corre una vez al dia:
    1. Decay de confianza en patrones sin uso
    2. Limpieza de patrones basura
    3. Desactivar respuestas malas
    4. Archivar intenciones vacias
    5. Clustering de frases no entendidas
    6. Backup de la base de datos
    """
    print("[mantenimiento] Iniciando mantenimiento diario...")

    # Decay
    afectados = bd.decay_confianza_inactivos(dias=30, factor=0.95)
    if afectados:
        print(f"  [decay] {afectados} patrones con confianza reducida")

    # Limpieza
    eliminados = bd.limpiar_patrones_basura(min_confianza=0.1, dias=60)
    if eliminados:
        print(f"  [limpieza] {eliminados} patrones basura eliminados")

    desactivadas = bd.desactivar_respuestas_malas(min_peso=0.2)
    if desactivadas:
        print(f"  [limpieza] {desactivadas} respuestas desactivadas")

    vacias = bd.archivar_intenciones_vacias()
    if vacias:
        print(f"  [limpieza] {vacias} intenciones vacias archivadas")

    # Clustering
    nuevos_tags = clustering_frases_no_entendidas()
    if nuevos_tags:
        print(f"  [clustering] {nuevos_tags} tags nuevos auto-creados")

    # Backup DB
    bd.backup_db()
    print("[mantenimiento] Completado.")

    return {"decay": afectados, "eliminados": eliminados, "desactivadas": desactivadas,
            "vacias": vacias, "nuevos_tags": nuevos_tags}


# ============================================================
#  FLUSH Y RE-ENTRENAMIENTO
# ============================================================

def flush_buffer():
    global _entrenamiento_pendiente
    with _lock_buffer:
        if not _buffer_patrones:
            return 0
        patrones = list(_buffer_patrones)
        _buffer_patrones.clear()
        _entrenamiento_pendiente = False
    guardados = 0
    for p in patrones:
        if bd.agregar_patron(p["tag"], p["texto"], p["origen"], p["confianza"]):
            guardados += 1
    return guardados


def hay_entrenamiento_pendiente():
    with _lock_buffer:
        return _entrenamiento_pendiente or len(_buffer_patrones) >= MAX_BUFFER_PATRONES


# ============================================================
#  SCHEDULER
# ============================================================

_scheduler_activo = False

def iniciar_scheduler():
    global _scheduler_activo, _ultimo_mantenimiento
    if _scheduler_activo:
        return
    _scheduler_activo = True
    _ultimo_mantenimiento = time.time()

    def loop():
        global _ultimo_mantenimiento
        while _scheduler_activo:
            time.sleep(INTERVALO_REENTRENAMIENTO)

            # Flush + re-entrenar si hay cambios
            guardados = flush_buffer()
            if guardados > 0 and _callback_reentrenar:
                try:
                    print(f"[auto-aprendizaje] {guardados} patrones nuevos. Re-entrenando...")
                    _callback_reentrenar()
                except Exception as e:
                    print(f"[auto-aprendizaje] Error: {e}")

            # Mantenimiento diario
            if time.time() - _ultimo_mantenimiento > INTERVALO_MANTENIMIENTO:
                try:
                    ejecutar_mantenimiento()
                    _ultimo_mantenimiento = time.time()
                    if _callback_reentrenar:
                        _callback_reentrenar()
                except Exception as e:
                    print(f"[mantenimiento] Error: {e}")

    thread = threading.Thread(target=loop, daemon=True, name="scheduler")
    thread.start()
    print("  Scheduler activo (re-entrenamiento cada 5 min, mantenimiento diario).")


def detener_scheduler():
    global _scheduler_activo
    _scheduler_activo = False


def forzar_reentrenamiento():
    guardados = flush_buffer()
    if _callback_reentrenar:
        _callback_reentrenar()
    return guardados
