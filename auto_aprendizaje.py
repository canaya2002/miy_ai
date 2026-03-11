"""
Auto-aprendizaje nivel maximo.
Incluye: refuerzo, contexto, feedback, clustering colectivo,
absorcion de respuestas, variaciones automaticas, sinonimos por
co-ocurrencia, fusion de intenciones, analisis de flujo.
Buffer de patrones via tabla staging en SQLite.
Sesiones con TTL para evitar memory leaks.
"""

import threading
import time
import re
import random
from collections import Counter
import base_datos as bd
from entrenar import limpiar, similitud_palabras

# ============================================================
#  CONFIGURACION
# ============================================================

UMBRAL_INMEDIATO = 0.90
UMBRAL_ALTO = 0.80
UMBRAL_MEDIO = 0.60
UMBRAL_BAJO = 0.30
UMBRAL_CONTEXTO = 0.50

MIN_SESIONES_CLUSTERING = 3
FUSION_UMBRAL_AUTO = 0.85
FUSION_UMBRAL_LOG = 0.70

PALABRAS_POSITIVAS = {
    "jaja", "jajaja", "jajajaja", "si", "exacto", "buena", "gracias",
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

# Templates para variaciones de respuestas
SALUDOS_TEMPLATE = ["Hola!", "Hey!", "Que onda!", "Que tal!", "Buenas!"]
CIERRES_TEMPLATE = ["Como estas?", "Que tal todo?", "Como te va?", "Como andas?", "Todo bien?"]

# ============================================================
#  SESIONES EN MEMORIA (con TTL y limpieza)
# ============================================================

_sesiones = {}
_sesiones_ts = {}
_lock_sesiones = threading.Lock()
MAX_SESIONES = 20000
SESION_TTL = 3600


def _limpiar_sesiones_expiradas():
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
                "ultima_respuesta_id": None,
                "historial_tags": []
            }
        _sesiones_ts[sesion_id] = time.time()
        return _sesiones[sesion_id].copy()


def actualizar_sesion(sesion_id, **kwargs):
    with _lock_sesiones:
        if sesion_id not in _sesiones:
            _sesiones[sesion_id] = {
                "frase_no_entendida": None,
                "ultimo_tag": None,
                "ultima_respuesta_id": None,
                "historial_tags": []
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

    # Confianza > 90%: agregar INMEDIATAMENTE como patron
    if confianza > UMBRAL_INMEDIATO and tag_detectado:
        bd.agregar_patron(tag_detectado, texto_usuario, origen="auto_refuerzo", confianza=0.9)
        _confirmar_staging(tag_detectado, sesion_id)
        _resolver_no_entendida(sesion_id, estado, tag_detectado)
        _registrar_tag_historial(sesion_id, tag_detectado)
        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "refuerzo_inmediato"

    # Confianza 80-90%: agregar a staging con confianza 0.7
    if confianza > UMBRAL_ALTO and tag_detectado:
        _agregar_patron_staging(tag_detectado, texto_usuario, "auto_refuerzo", 0.7)
        _confirmar_staging(tag_detectado, sesion_id)
        _resolver_no_entendida(sesion_id, estado, tag_detectado)
        _registrar_tag_historial(sesion_id, tag_detectado)
        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "refuerzo_staging"

    # Confianza 60-80%: responder normalmente, no agregar nada
    if confianza > UMBRAL_MEDIO and tag_detectado:
        _confirmar_staging(tag_detectado, sesion_id)
        _registrar_tag_historial(sesion_id, tag_detectado)
        actualizar_sesion(sesion_id, ultimo_tag=tag_detectado, ultima_respuesta_id=respuesta_id)
        return "respuesta_normal"

    # Confianza 30-60%: intentar inferir por flujo conversacional
    if confianza > UMBRAL_BAJO and tag_detectado:
        tag_inferido = _inferir_por_flujo(sesion_id, tag_detectado)
        if tag_inferido:
            _agregar_patron_staging(tag_inferido, texto_usuario, "auto_flujo", 0.5)
        actualizar_sesion(sesion_id, frase_no_entendida=texto_usuario, ultima_respuesta_id=None)
        return "baja_confianza"

    # Confianza < 30%: guardar en desconocidos
    bd.guardar_desconocido(texto_usuario, sesion_id)
    actualizar_sesion(sesion_id, frase_no_entendida=texto_usuario, ultima_respuesta_id=None)
    return "no_entendido"


def _es_feedback_positivo(t):
    return t in PALABRAS_POSITIVAS


def _es_feedback_negativo(t):
    return t in PALABRAS_NEGATIVAS


def _agregar_patron_staging(tag, texto, origen, confianza):
    bd.agregar_patron_pendiente(tag, texto, origen, confianza)


def _confirmar_staging(tag, sesion_id):
    """Confirma patrones pendientes de este tag si la sesion es diferente."""
    bd.confirmar_patron_pendiente(tag, sesion_id)


def _resolver_no_entendida(sesion_id, estado, tag_detectado):
    """Si habia una frase no entendida, asignarla al tag actual."""
    if estado["frase_no_entendida"]:
        _agregar_patron_staging(tag_detectado, estado["frase_no_entendida"],
                                "auto_contexto", UMBRAL_CONTEXTO)
        actualizar_sesion(sesion_id, frase_no_entendida=None)


def _registrar_tag_historial(sesion_id, tag):
    """Registra el tag en el historial de la sesion (ultimos 10)."""
    with _lock_sesiones:
        if sesion_id in _sesiones:
            hist = _sesiones[sesion_id].get("historial_tags", [])
            hist.append(tag)
            if len(hist) > 10:
                hist = hist[-10:]
            _sesiones[sesion_id]["historial_tags"] = hist


def _inferir_por_flujo(sesion_id, tag_detectado):
    """
    Analiza los ultimos 5 mensajes de la sesion.
    Si la mayoria comparten un tag, asigna el mensaje actual a ese tag.
    """
    mensajes = bd.obtener_ultimos_mensajes_sesion(sesion_id, limite=5)
    if len(mensajes) < 3:
        return None

    tags_vistos = [m["tag"] for m in mensajes if m["tag"] and m["confianza"] and m["confianza"] > 0.6]
    if len(tags_vistos) < 2:
        return None

    conteo = Counter(tags_vistos)
    tag_mas_comun, freq = conteo.most_common(1)[0]

    # Si 3+ de los ultimos mensajes son del mismo tag
    if freq >= 3 and tag_mas_comun != tag_detectado:
        return tag_mas_comun

    return None


# ============================================================
#  GENERACION AUTOMATICA DE VARIACIONES
# ============================================================

def generar_variaciones_respuestas():
    """
    Para intenciones con pocas respuestas, genera variaciones
    combinando partes de respuestas existentes.
    """
    intenciones = bd.obtener_intenciones_para_entrenamiento()
    total_generadas = 0

    for intencion in intenciones:
        respuestas = [r for r in intencion["respuestas"] if r["peso"] > 0.7]
        if len(respuestas) >= 4:
            continue

        textos = [r["texto"] for r in respuestas]
        if len(textos) < 1:
            continue

        # Intentar generar variaciones con shuffling de partes
        nuevas = _generar_variaciones_simples(textos)
        for nueva in nuevas[:3]:
            if bd.agregar_respuesta(intencion["tag"], nueva, peso=0.7):
                total_generadas += 1

    return total_generadas


def _generar_variaciones_simples(textos):
    """Genera variaciones mezclando partes de respuestas existentes."""
    variaciones = []

    # Separar en inicio y fin por el primer signo de puntuacion o espacio largo
    partes_inicio = []
    partes_fin = []

    for t in textos:
        # Buscar punto de corte: !, ?, .
        for sep in ["!", "?", "."]:
            idx = t.find(sep)
            if idx > 0 and idx < len(t) - 1:
                partes_inicio.append(t[:idx+1].strip())
                resto = t[idx+1:].strip()
                if resto:
                    partes_fin.append(resto)
                break
        else:
            # Sin puntuacion, buscar corte por coma
            idx = t.find(",")
            if idx > 0:
                partes_inicio.append(t[:idx+1].strip())
                resto = t[idx+1:].strip()
                if resto:
                    partes_fin.append(resto)

    if not partes_inicio or not partes_fin:
        return []

    # Combinar
    for inicio in partes_inicio:
        for fin in partes_fin:
            nueva = inicio + " " + fin
            if nueva not in textos and len(nueva) < 200:
                variaciones.append(nueva)

    return variaciones


# ============================================================
#  DETECCION DE SINONIMOS POR CO-OCURRENCIA
# ============================================================

def detectar_sinonimos_coocurrencia():
    """
    Analiza palabras que aparecen frecuentemente en patrones del mismo tag.
    Si dos palabras aparecen siempre juntas en el mismo tag pero nunca en la misma frase,
    probablemente son sinonimos.
    """
    palabras_por_tag = bd.obtener_palabras_por_tag()
    if not palabras_por_tag:
        return 0

    # Para cada tag, obtener las palabras de cada patron individual
    intenciones = bd.obtener_intenciones_para_entrenamiento()
    palabra_tags = {}  # palabra -> set de tags

    for intencion in intenciones:
        tag = intencion["tag"]
        for patron in intencion["patrones"]:
            for palabra in limpiar(patron):
                if len(palabra) > 2 and palabra not in STOP_WORDS:
                    if palabra not in palabra_tags:
                        palabra_tags[palabra] = set()
                    palabra_tags[palabra].add(tag)

    # Buscar pares de palabras que comparten muchos tags
    palabras = list(palabra_tags.keys())
    nuevos = 0

    for i in range(len(palabras)):
        for j in range(i + 1, len(palabras)):
            p1, p2 = palabras[i], palabras[j]
            tags_comunes = palabra_tags[p1] & palabra_tags[p2]
            tags_total = palabra_tags[p1] | palabra_tags[p2]

            # Si comparten > 70% de tags y ambas aparecen en > 1 tag
            if (len(tags_comunes) >= 2 and
                    len(tags_comunes) / len(tags_total) > 0.7 and
                    similitud_palabras(p1, p2) < 0.5):  # no son la misma palabra con typo
                bd.agregar_sinonimo(p1, p2)
                nuevos += 1

    return nuevos


# ============================================================
#  FUSION AUTOMATICA DE INTENCIONES SIMILARES
# ============================================================

def evaluar_fusion_intenciones():
    """
    Detecta tags con > 70% overlap de palabras y los fusiona
    (automaticamente si > 85%, log si 70-85%).
    """
    palabras_por_tag = bd.obtener_palabras_por_tag()
    tags = list(palabras_por_tag.keys())
    fusiones = 0

    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            t1, t2 = tags[i], tags[j]
            w1, w2 = palabras_por_tag.get(t1, set()), palabras_por_tag.get(t2, set())
            if not w1 or not w2:
                continue

            shared = w1 & w2
            total = w1 | w2
            overlap = len(shared) / len(total) if total else 0

            if overlap >= FUSION_UMBRAL_AUTO:
                # Determinar cual tiene mas patrones
                p1 = bd.obtener_patrones_de_tag(t1)
                p2 = bd.obtener_patrones_de_tag(t2)
                if len(p1) >= len(p2):
                    mantener, eliminar = t1, t2
                else:
                    mantener, eliminar = t2, t1
                if bd.fusionar_intenciones(mantener, eliminar):
                    print(f"  [fusion] Auto-fusion: '{eliminar}' absorbido por '{mantener}' ({overlap:.0%} overlap)")
                    fusiones += 1
            elif overlap >= FUSION_UMBRAL_LOG:
                print(f"  [fusion] Posible fusion: '{t1}' y '{t2}' ({overlap:.0%} overlap)")

    return fusiones


# ============================================================
#  CLUSTERING DE FRASES NO ENTENDIDAS
# ============================================================

def clustering_frases_no_entendidas():
    frases = bd.obtener_frases_no_entendidas(dias=7)
    desconocidos = bd.obtener_desconocidos(dias=7)
    todas = frases + desconocidos

    if len(todas) < MIN_SESIONES_CLUSTERING:
        return 0

    grupos = {}
    for f in todas:
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

    renombrados = bd.renombrar_tags_confusos()
    if renombrados:
        print(f"  [limpieza] {renombrados} tags confusos renombrados")

    # Deduplicar patrones similares
    duplicados = bd.detectar_patrones_duplicados(umbral=0.9)
    if duplicados:
        print(f"  [dedup] {duplicados} patrones duplicados eliminados")

    # Promover patrones confirmados por multiples sesiones
    promovidos = bd.promover_patrones_confirmados(min_confirmaciones=2)
    if promovidos:
        print(f"  [staging] {promovidos} patrones promovidos por confirmacion multiple")

    # Limpiar staging viejo
    staging_limpio = bd.limpiar_staging_viejo(dias=14)
    if staging_limpio:
        print(f"  [staging] {staging_limpio} patrones pendientes viejos eliminados")

    # Clustering
    nuevos_tags = clustering_frases_no_entendidas()
    if nuevos_tags:
        print(f"  [clustering] {nuevos_tags} tags nuevos auto-creados")

    # Sinonimos
    nuevos_sin = detectar_sinonimos_coocurrencia()
    if nuevos_sin:
        print(f"  [sinonimos] {nuevos_sin} nuevos pares de sinonimos detectados")

    # Fusion
    fusiones = evaluar_fusion_intenciones()
    if fusiones:
        print(f"  [fusion] {fusiones} intenciones fusionadas")

    # Variaciones de respuestas
    variaciones = generar_variaciones_respuestas()
    if variaciones:
        print(f"  [variaciones] {variaciones} variaciones de respuestas generadas")

    # Purga
    eliminados_msg = bd.purgar_mensajes_viejos(dias=30)
    if eliminados_msg:
        print(f"  [purga] {eliminados_msg} mensajes viejos eliminados")

    bd.backup_db()
    bd.exportar_log_semanal()
    print("[mantenimiento] Completado.")

    return {"decay": afectados, "eliminados": eliminados, "desactivadas": desactivadas,
            "vacias": vacias, "nuevos_tags": nuevos_tags, "fusiones": fusiones,
            "variaciones": variaciones, "mensajes_purgados": eliminados_msg}
