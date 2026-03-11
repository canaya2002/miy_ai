"""
Sistema de auto-aprendizaje.
La IA aprende automaticamente de cada conversacion sin intervencion humana.

Mecanismos:
1. Refuerzo automatico: si entiende con alta confianza (>85%), guarda la frase como patron nuevo
2. Inferencia por contexto: si no entiende algo pero lo siguiente si, asocia ambas frases al mismo tema
3. Feedback implicito: si el usuario reacciona bien ("jaja", "exacto"), sube el peso de esa respuesta
4. Re-entrenamiento periodico: cada 5 min o cada 20 patrones nuevos, en un thread de background
"""

import threading
import time
import base_datos as bd
from entrenar import limpiar

# ============================================================
#  CONFIGURACION
# ============================================================

# Umbrales de confianza para decidir que hacer
UMBRAL_ALTO = 0.85       # Arriba de esto: refuerzo automatico
UMBRAL_MEDIO = 0.60      # Arriba de esto: responde pero no aprende
UMBRAL_CONTEXTO = 0.50   # Confianza asignada a patrones inferidos por contexto

# Re-entrenamiento automatico
INTERVALO_REENTRENAMIENTO = 300  # 5 minutos en segundos
MAX_BUFFER_PATRONES = 20         # Re-entrenar cuando hay 20+ patrones acumulados

# Palabras que indican feedback positivo o negativo
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


# ============================================================
#  ESTADO EN MEMORIA (por sesion, no persistente)
# ============================================================

# Buffer de patrones nuevos pendientes de guardarse en SQLite
_buffer_patrones = []
_lock_buffer = threading.Lock()

# Estado de cada sesion activa
# sesion_id -> {frase_no_entendida, ultimo_tag, ultima_respuesta_id}
_sesiones = {}
_lock_sesiones = threading.Lock()

# Callback que el servidor registra para disparar re-entrenamiento
_callback_reentrenar = None

# Flag de entrenamiento pendiente
_entrenamiento_pendiente = False


def registrar_callback_reentrenar(callback):
    """El servidor registra aqui su funcion de re-entrenamiento"""
    global _callback_reentrenar
    _callback_reentrenar = callback


# ============================================================
#  TRACKING DE SESIONES
# ============================================================

def obtener_estado_sesion(sesion_id):
    """Obtiene el estado actual de una sesion (lo crea si no existe)"""
    with _lock_sesiones:
        if sesion_id not in _sesiones:
            _sesiones[sesion_id] = {
                "frase_no_entendida": None,
                "ultimo_tag": None,
                "ultima_respuesta_id": None
            }
        return _sesiones[sesion_id].copy()


def actualizar_sesion(sesion_id, **kwargs):
    """Actualiza campos del estado de una sesion"""
    with _lock_sesiones:
        if sesion_id not in _sesiones:
            _sesiones[sesion_id] = {
                "frase_no_entendida": None,
                "ultimo_tag": None,
                "ultima_respuesta_id": None
            }
        _sesiones[sesion_id].update(kwargs)


# ============================================================
#  LOGICA DE AUTO-APRENDIZAJE
# ============================================================

def procesar_mensaje(sesion_id, texto_usuario, tag_detectado, confianza, respuesta_id=None):
    """
    Analiza un mensaje y decide si la IA debe aprender algo.
    Se llama DESPUES de que la IA genera su respuesta.

    Retorna una string indicando que paso:
    - "feedback_positivo": el usuario reacciono bien a la respuesta anterior
    - "feedback_negativo": el usuario reacciono mal
    - "refuerzo": se reforzó un patron con alta confianza
    - "respuesta_normal": respondio bien pero sin aprender
    - "no_entendido": no entendio, guardó para inferencia por contexto
    """
    estado = obtener_estado_sesion(sesion_id)
    texto_limpio = " ".join(limpiar(texto_usuario))

    # --- CASO 1: Detectar feedback del usuario sobre la respuesta anterior ---
    if estado["ultima_respuesta_id"]:
        if _es_feedback_positivo(texto_limpio):
            bd.ajustar_peso_respuesta(estado["ultima_respuesta_id"], 0.15)
            actualizar_sesion(sesion_id, ultima_respuesta_id=None)
            return "feedback_positivo"

        if _es_feedback_negativo(texto_limpio):
            bd.ajustar_peso_respuesta(estado["ultima_respuesta_id"], -0.15)
            actualizar_sesion(sesion_id, ultima_respuesta_id=None)
            return "feedback_negativo"

    # --- CASO 2: Refuerzo automatico (confianza alta > 85%) ---
    if confianza > UMBRAL_ALTO and tag_detectado:
        _agregar_patron_buffer(tag_detectado, texto_usuario, "auto_refuerzo", confianza)

        # Si habia una frase no entendida antes, inferir que es del mismo tema
        if estado["frase_no_entendida"]:
            _agregar_patron_buffer(
                tag_detectado,
                estado["frase_no_entendida"],
                "auto_contexto",
                UMBRAL_CONTEXTO
            )
            actualizar_sesion(sesion_id, frase_no_entendida=None)

        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "refuerzo"

    # --- CASO 3: Confianza media (60-85%) - responde sin aprender ---
    if confianza > UMBRAL_MEDIO and tag_detectado:
        # Si habia frase no entendida, asociar con confianza baja
        if estado["frase_no_entendida"]:
            _agregar_patron_buffer(
                tag_detectado,
                estado["frase_no_entendida"],
                "auto_contexto",
                UMBRAL_CONTEXTO * 0.7
            )
            actualizar_sesion(sesion_id, frase_no_entendida=None)

        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "respuesta_normal"

    # --- CASO 4: No entendio (<60%) ---
    actualizar_sesion(sesion_id, frase_no_entendida=texto_usuario, ultima_respuesta_id=None)
    return "no_entendido"


def _es_feedback_positivo(texto_limpio):
    """Detecta si el mensaje es una reaccion positiva"""
    return texto_limpio in PALABRAS_POSITIVAS


def _es_feedback_negativo(texto_limpio):
    """Detecta si el mensaje es una reaccion negativa"""
    return texto_limpio in PALABRAS_NEGATIVAS


def _agregar_patron_buffer(tag, texto, origen, confianza):
    """Agrega un patron al buffer de pendientes (se guardara en el proximo flush)"""
    global _entrenamiento_pendiente
    with _lock_buffer:
        _buffer_patrones.append({
            "tag": tag,
            "texto": texto,
            "origen": origen,
            "confianza": confianza
        })

        if len(_buffer_patrones) >= MAX_BUFFER_PATRONES:
            _entrenamiento_pendiente = True


# ============================================================
#  FLUSH Y RE-ENTRENAMIENTO
# ============================================================

def flush_buffer():
    """
    Vacia el buffer: guarda los patrones pendientes en SQLite.
    Devuelve el numero de patrones guardados.
    """
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
    """Indica si hay suficientes patrones nuevos para justificar re-entrenamiento"""
    with _lock_buffer:
        return _entrenamiento_pendiente or len(_buffer_patrones) >= MAX_BUFFER_PATRONES


# ============================================================
#  SCHEDULER DE RE-ENTRENAMIENTO PERIODICO
# ============================================================

_scheduler_activo = False


def iniciar_scheduler():
    """
    Inicia un thread en background que cada 5 minutos:
    1. Vacia el buffer de patrones nuevos a SQLite
    2. Si hubo cambios, dispara re-entrenamiento
    """
    global _scheduler_activo
    if _scheduler_activo:
        return
    _scheduler_activo = True

    def loop():
        while _scheduler_activo:
            time.sleep(INTERVALO_REENTRENAMIENTO)

            guardados = flush_buffer()

            if guardados > 0 and _callback_reentrenar:
                try:
                    print(f"[auto-aprendizaje] {guardados} patrones nuevos. Re-entrenando...")
                    _callback_reentrenar()
                except Exception as e:
                    print(f"[auto-aprendizaje] Error al re-entrenar: {e}")

    thread = threading.Thread(target=loop, daemon=True, name="scheduler-reentrenamiento")
    thread.start()
    print("  Scheduler de re-entrenamiento activo (cada 5 min).")


def detener_scheduler():
    """Detiene el scheduler"""
    global _scheduler_activo
    _scheduler_activo = False


def forzar_reentrenamiento():
    """Fuerza flush + re-entrenamiento inmediato"""
    guardados = flush_buffer()
    if _callback_reentrenar:
        _callback_reentrenar()
    return guardados
