"""
Base de datos SQLite para la IA.
Maneja intenciones, patrones, respuestas, mensajes, estadisticas,
sinonimos, hiperparametros, desconocidos.
Optimizado: WAL, busy_timeout, write batching, connection pooling, indices.
"""

import sqlite3
import os
import threading
import shutil
import time as _time
import queue
import json

# ============================================================
#  RUTAS DE DATOS
# ============================================================

def obtener_ruta_datos():
    if os.path.exists("/data"):
        return "/data"
    return os.path.dirname(os.path.abspath(__file__))

RUTA_DATOS = obtener_ruta_datos()
RUTA_DB = os.path.join(RUTA_DATOS, "ia.db")
RUTA_MODELO = os.path.join(RUTA_DATOS, "ia_entrenada.pth")
RUTA_BACKUP_MODELO = os.path.join(RUTA_DATOS, "ia_backup.pth")
RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos.json")
RUTA_LOG = os.path.join(RUTA_DATOS, "ia_semanal.log")

# ============================================================
#  LIMITES DE SEGURIDAD
# ============================================================

MAX_INTENCIONES = 1000
MAX_PATRONES_POR_INTENCION = 100
MAX_RESPUESTAS_POR_INTENCION = 50
MAX_DB_SIZE_MB = 500

# ============================================================
#  CONNECTION POOLING
# ============================================================

_pool = queue.Queue(maxsize=5)


class PooledConnection:
    """Wrapper que devuelve la conexion al pool en vez de cerrarla."""

    def __init__(self, conn):
        self._conn = conn

    def close(self):
        try:
            self._conn.rollback()
        except Exception:
            pass
        try:
            _pool.put_nowait(self._conn)
        except queue.Full:
            try:
                self._conn.close()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _crear_conexion_raw():
    conn = sqlite3.connect(RUTA_DB, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-8000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def conectar():
    try:
        raw = _pool.get_nowait()
        try:
            raw.execute("SELECT 1")
            return PooledConnection(raw)
        except Exception:
            try:
                raw.close()
            except Exception:
                pass
    except queue.Empty:
        pass
    return PooledConnection(_crear_conexion_raw())


# ============================================================
#  CREAR TABLAS
# ============================================================

def crear_tablas():
    conn = conectar()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS intenciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS patrones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intencion_id INTEGER NOT NULL REFERENCES intenciones(id),
            texto TEXT NOT NULL,
            origen TEXT DEFAULT 'manual',
            confianza REAL DEFAULT 1.0,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS respuestas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intencion_id INTEGER NOT NULL REFERENCES intenciones(id),
            texto TEXT NOT NULL,
            peso REAL DEFAULT 1.0,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS mensajes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sesion TEXT NOT NULL,
            rol TEXT NOT NULL,
            texto TEXT NOT NULL,
            tag_detectado TEXT,
            confianza REAL,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS estadisticas (
            clave TEXT PRIMARY KEY,
            valor INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS patrones_pendientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag TEXT NOT NULL,
            texto TEXT NOT NULL,
            origen TEXT DEFAULT 'auto',
            confianza REAL DEFAULT 0.5,
            confirmaciones INTEGER DEFAULT 0,
            sesiones_vistas TEXT DEFAULT '[]',
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sinonimos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            palabra_a TEXT NOT NULL,
            palabra_b TEXT NOT NULL,
            tag_id INTEGER,
            frecuencia INTEGER DEFAULT 1,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(palabra_a, palabra_b)
        );
        CREATE TABLE IF NOT EXISTS hiperparametros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lr REAL NOT NULL,
            epocas INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS desconocidos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT NOT NULL,
            sesion TEXT NOT NULL,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indices
        CREATE INDEX IF NOT EXISTS idx_mensajes_sesion ON mensajes(sesion);
        CREATE INDEX IF NOT EXISTS idx_mensajes_fecha ON mensajes(fecha);
        CREATE INDEX IF NOT EXISTS idx_mensajes_confianza ON mensajes(confianza);
        CREATE INDEX IF NOT EXISTS idx_patrones_intencion ON patrones(intencion_id);
        CREATE INDEX IF NOT EXISTS idx_respuestas_intencion ON respuestas(intencion_id);
        CREATE INDEX IF NOT EXISTS idx_desconocidos_fecha ON desconocidos(fecha);
        CREATE INDEX IF NOT EXISTS idx_sinonimos_palabra ON sinonimos(palabra_a);
        CREATE INDEX IF NOT EXISTS idx_pendientes_tag ON patrones_pendientes(tag);

        INSERT OR IGNORE INTO estadisticas (clave, valor) VALUES ('mensajes_totales', 0);
        INSERT OR IGNORE INTO estadisticas (clave, valor) VALUES ('sesiones_totales', 0);
        INSERT OR IGNORE INTO estadisticas (clave, valor) VALUES ('entrenamientos_totales', 0);
    """)
    conn.commit()
    conn.close()

    _migrar_columnas_pendientes()


def _migrar_columnas_pendientes():
    """Agrega columnas nuevas a patrones_pendientes si no existen."""
    conn = conectar()
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(patrones_pendientes)")]
        if "confirmaciones" not in cols:
            conn.execute("ALTER TABLE patrones_pendientes ADD COLUMN confirmaciones INTEGER DEFAULT 0")
        if "sesiones_vistas" not in cols:
            conn.execute("ALTER TABLE patrones_pendientes ADD COLUMN sesiones_vistas TEXT DEFAULT '[]'")
        conn.commit()
    except Exception:
        pass
    conn.close()


# ============================================================
#  LECTURA
# ============================================================

def obtener_intenciones_para_entrenamiento():
    conn = conectar()
    intenciones_map = {}
    for row in conn.execute("SELECT id, tag FROM intenciones ORDER BY id"):
        intenciones_map[row["id"]] = {"tag": row["tag"], "patrones": [], "respuestas": []}
    for p in conn.execute("SELECT intencion_id, texto FROM patrones"):
        if p["intencion_id"] in intenciones_map:
            intenciones_map[p["intencion_id"]]["patrones"].append(p["texto"])
    for r in conn.execute("SELECT intencion_id, id, texto, peso FROM respuestas"):
        if r["intencion_id"] in intenciones_map:
            intenciones_map[r["intencion_id"]]["respuestas"].append(
                {"texto": r["texto"], "peso": r["peso"], "id": r["id"]})
    conn.close()
    return [v for v in intenciones_map.values() if v["patrones"] and v["respuestas"]]


def obtener_respuestas_por_tag(tag):
    conn = conectar()
    rows = conn.execute("""
        SELECT r.id, r.texto, r.peso FROM respuestas r
        JOIN intenciones i ON r.intencion_id = i.id WHERE i.tag = ?
    """, (tag,)).fetchall()
    conn.close()
    return [{"id": r["id"], "texto": r["texto"], "peso": r["peso"]} for r in rows]


def obtener_estadisticas():
    conn = conectar()
    temas = conn.execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    patrones = conn.execute("SELECT COUNT(*) FROM patrones").fetchone()[0]
    respuestas = conn.execute("SELECT COUNT(*) FROM respuestas").fetchone()[0]
    mensajes_totales = conn.execute("SELECT valor FROM estadisticas WHERE clave='mensajes_totales'").fetchone()[0]
    sesiones_totales = conn.execute("SELECT valor FROM estadisticas WHERE clave='sesiones_totales'").fetchone()[0]
    entrenamientos = conn.execute("SELECT valor FROM estadisticas WHERE clave='entrenamientos_totales'").fetchone()[0]
    lista_temas = [r["tag"] for r in conn.execute("SELECT tag FROM intenciones ORDER BY tag")]
    conn.close()
    return {
        "temas": temas, "patrones": patrones, "respuestas": respuestas,
        "mensajes_totales": mensajes_totales, "sesiones_totales": sesiones_totales,
        "entrenamientos": entrenamientos, "lista_temas": lista_temas
    }


def obtener_temas_detallados():
    conn = conectar()
    temas = []
    for row in conn.execute("SELECT id, tag FROM intenciones ORDER BY tag"):
        np_count = conn.execute("SELECT COUNT(*) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        nr = conn.execute("SELECT COUNT(*) FROM respuestas WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        fecha = conn.execute("SELECT MIN(fecha) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        origenes = {}
        for o in conn.execute("SELECT origen, COUNT(*) as c FROM patrones WHERE intencion_id=? GROUP BY origen", (row["id"],)):
            origenes[o["origen"]] = o["c"]
        temas.append({"tag": row["tag"], "patrones": np_count, "respuestas": nr,
                       "fecha_creacion": fecha, "origenes": origenes})
    conn.close()
    return temas


def obtener_palabras_por_tag():
    conn = conectar()
    resultado = {}
    for row in conn.execute("SELECT id, tag FROM intenciones"):
        patrones = conn.execute("SELECT texto FROM patrones WHERE intencion_id=?", (row["id"],)).fetchall()
        palabras = set()
        for p in patrones:
            for palabra in p["texto"].lower().split():
                if len(palabra) > 2:
                    palabras.add(palabra)
        resultado[row["tag"]] = palabras
    conn.close()
    return resultado


def obtener_frases_no_entendidas(dias=7):
    conn = conectar()
    rows = conn.execute("""
        SELECT texto, sesion FROM mensajes
        WHERE rol='usuario' AND (confianza IS NULL OR confianza < 0.6)
        AND fecha > datetime('now', ?)
    """, (f'-{dias} days',)).fetchall()
    conn.close()
    return [{"texto": r["texto"], "sesion": r["sesion"]} for r in rows]


def tags_activos_en_periodo(dias=30):
    conn = conectar()
    rows = conn.execute("""
        SELECT DISTINCT tag_detectado FROM mensajes
        WHERE tag_detectado IS NOT NULL AND confianza > 0.6
        AND fecha > datetime('now', ?)
    """, (f'-{dias} days',)).fetchall()
    conn.close()
    return {r[0] for r in rows}


def obtener_ultimos_mensajes_sesion(sesion_id, limite=5):
    """Obtiene los ultimos N mensajes de una sesion para analisis de flujo."""
    conn = conectar()
    rows = conn.execute("""
        SELECT texto, rol, tag_detectado, confianza FROM mensajes
        WHERE sesion=? ORDER BY fecha DESC LIMIT ?
    """, (sesion_id, limite)).fetchall()
    conn.close()
    return [{"texto": r["texto"], "rol": r["rol"], "tag": r["tag_detectado"],
             "confianza": r["confianza"]} for r in reversed(rows)]


def obtener_patrones_de_tag(tag):
    """Obtiene todos los patrones de un tag."""
    conn = conectar()
    row = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
    if not row:
        conn.close()
        return []
    patrones = conn.execute("SELECT texto, origen, confianza FROM patrones WHERE intencion_id=?",
                            (row["id"],)).fetchall()
    conn.close()
    return [{"texto": p["texto"], "origen": p["origen"], "confianza": p["confianza"]} for p in patrones]


# ============================================================
#  SINONIMOS
# ============================================================

def obtener_sinonimos():
    """Devuelve dict {palabra: palabra_canonica} para reemplazo."""
    conn = conectar()
    rows = conn.execute("SELECT palabra_a, palabra_b, frecuencia FROM sinonimos ORDER BY frecuencia DESC").fetchall()
    conn.close()
    mapa = {}
    for r in rows:
        a, b = r["palabra_a"], r["palabra_b"]
        if a not in mapa and b not in mapa:
            mapa[b] = a
        elif a in mapa and b not in mapa:
            mapa[b] = mapa.get(a, a)
        elif b in mapa and a not in mapa:
            mapa[a] = mapa.get(b, b)
    return mapa


def agregar_sinonimo(palabra_a, palabra_b, tag_id=None):
    if palabra_a == palabra_b:
        return
    a, b = sorted([palabra_a.lower(), palabra_b.lower()])
    conn = conectar()
    existing = conn.execute("SELECT id, frecuencia FROM sinonimos WHERE palabra_a=? AND palabra_b=?", (a, b)).fetchone()
    if existing:
        conn.execute("UPDATE sinonimos SET frecuencia=frecuencia+1 WHERE id=?", (existing["id"],))
    else:
        conn.execute("INSERT INTO sinonimos (palabra_a, palabra_b, tag_id) VALUES (?,?,?)", (a, b, tag_id))
    conn.commit()
    conn.close()


# ============================================================
#  HIPERPARAMETROS
# ============================================================

def guardar_hiperparametros(lr, epocas, batch_size, accuracy):
    conn = conectar()
    conn.execute("INSERT INTO hiperparametros (lr, epocas, batch_size, accuracy) VALUES (?,?,?,?)",
                 (lr, epocas, batch_size, accuracy))
    conn.commit()
    conn.close()


def obtener_mejores_hiperparametros():
    conn = conectar()
    row = conn.execute("""
        SELECT lr, epocas, batch_size, accuracy FROM hiperparametros
        ORDER BY accuracy DESC LIMIT 1
    """).fetchone()
    conn.close()
    if row:
        return {"lr": row["lr"], "epocas": row["epocas"],
                "batch_size": row["batch_size"], "accuracy": row["accuracy"]}
    return None


# ============================================================
#  DESCONOCIDOS
# ============================================================

def guardar_desconocido(texto, sesion):
    conn = conectar()
    conn.execute("INSERT INTO desconocidos (texto, sesion) VALUES (?,?)", (texto, sesion))
    conn.commit()
    conn.close()


def obtener_desconocidos(dias=7):
    conn = conectar()
    rows = conn.execute("""
        SELECT texto, sesion FROM desconocidos WHERE fecha > datetime('now', ?)
    """, (f'-{dias} days',)).fetchall()
    conn.close()
    return [{"texto": r["texto"], "sesion": r["sesion"]} for r in rows]


# ============================================================
#  CAMBIOS DESDE TIMESTAMP (para bienvenida)
# ============================================================

def obtener_cambios_desde(timestamp_iso):
    """Devuelve nuevos temas y patrones desde una fecha ISO."""
    conn = conectar()
    nuevos_temas = conn.execute("""
        SELECT COUNT(DISTINCT i.id) FROM intenciones i
        JOIN patrones p ON p.intencion_id = i.id
        WHERE p.fecha > ?
        AND NOT EXISTS (SELECT 1 FROM patrones p2 WHERE p2.intencion_id = i.id AND p2.fecha <= ?)
    """, (timestamp_iso, timestamp_iso)).fetchone()[0]

    nuevos_patrones = conn.execute("""
        SELECT COUNT(*) FROM patrones WHERE fecha > ?
    """, (timestamp_iso,)).fetchone()[0]

    temas_recientes = [r["tag"] for r in conn.execute("""
        SELECT DISTINCT i.tag FROM intenciones i
        JOIN patrones p ON p.intencion_id = i.id
        WHERE p.fecha > ? ORDER BY p.fecha DESC LIMIT 10
    """, (timestamp_iso,))]

    conn.close()
    return {
        "nuevos_temas": nuevos_temas,
        "nuevos_patrones": nuevos_patrones,
        "temas_recientes": temas_recientes
    }


# ============================================================
#  ESCRITURA
# ============================================================

def agregar_intencion(tag):
    conn = conectar()
    try:
        conn.execute("INSERT INTO intenciones (tag) VALUES (?)", (tag,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    id_tag = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()[0]
    conn.close()
    return id_tag


def agregar_patron(tag, texto, origen="manual", confianza=1.0):
    conn = conectar()
    row = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
    if not row:
        conn.close()
        return False
    existe = conn.execute("SELECT id FROM patrones WHERE intencion_id=? AND texto=?", (row["id"], texto)).fetchone()
    if not existe:
        count = conn.execute("SELECT COUNT(*) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        if count >= MAX_PATRONES_POR_INTENCION:
            conn.close()
            return False
        conn.execute("INSERT INTO patrones (intencion_id,texto,origen,confianza) VALUES (?,?,?,?)",
                     (row["id"], texto, origen, confianza))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


def agregar_respuesta(tag, texto, peso=1.0):
    conn = conectar()
    row = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
    if not row:
        conn.close()
        return False
    existe = conn.execute("SELECT id FROM respuestas WHERE intencion_id=? AND texto=?", (row["id"], texto)).fetchone()
    if not existe:
        count = conn.execute("SELECT COUNT(*) FROM respuestas WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        if count >= MAX_RESPUESTAS_POR_INTENCION:
            conn.close()
            return False
        conn.execute("INSERT INTO respuestas (intencion_id,texto,peso) VALUES (?,?,?)", (row["id"], texto, peso))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


def agregar_patron_pendiente(tag, texto, origen="auto", confianza=0.5):
    conn = conectar()
    conn.execute("INSERT INTO patrones_pendientes (tag,texto,origen,confianza) VALUES (?,?,?,?)",
                 (tag, texto, origen, confianza))
    conn.commit()
    conn.close()


def ajustar_peso_respuesta(respuesta_id, delta):
    conn = conectar()
    conn.execute("UPDATE respuestas SET peso=MAX(0.1, peso+?) WHERE id=?", (delta, respuesta_id))
    conn.commit()
    conn.close()


# ============================================================
#  STAGING: CONFIRMACION MULTI-SESION
# ============================================================

def confirmar_patron_pendiente(tag, sesion_id):
    """Incrementa confirmaciones si esta sesion no lo habia confirmado."""
    conn = conectar()
    pendientes = conn.execute(
        "SELECT id, sesiones_vistas, confirmaciones FROM patrones_pendientes WHERE tag=?", (tag,)).fetchall()
    promovidos = 0
    for p in pendientes:
        try:
            vistas = json.loads(p["sesiones_vistas"] or "[]")
        except Exception:
            vistas = []
        if sesion_id in vistas:
            continue
        vistas.append(sesion_id)
        nuevas_conf = p["confirmaciones"] + 1
        conn.execute("UPDATE patrones_pendientes SET confirmaciones=?, sesiones_vistas=? WHERE id=?",
                     (nuevas_conf, json.dumps(vistas), p["id"]))
    conn.commit()
    conn.close()
    return promovidos


def promover_patrones_confirmados(min_confirmaciones=2):
    """Promueve patrones staging con suficientes confirmaciones."""
    conn = conectar()
    listos = conn.execute(
        "SELECT id, tag, texto, origen, confianza FROM patrones_pendientes WHERE confirmaciones >= ?",
        (min_confirmaciones,)).fetchall()
    promovidos = 0
    ids_borrar = []
    for p in listos:
        ids_borrar.append(p["id"])
        agregar_patron(p["tag"], p["texto"], origen=p["origen"], confianza=0.8)
        promovidos += 1
    if ids_borrar:
        placeholders = ",".join("?" * len(ids_borrar))
        conn.execute(f"DELETE FROM patrones_pendientes WHERE id IN ({placeholders})", ids_borrar)
        conn.commit()
    conn.close()
    return promovidos


def limpiar_staging_viejo(dias=14):
    """Elimina patrones pendientes viejos sin confirmaciones."""
    conn = conectar()
    eliminados = conn.execute("""
        DELETE FROM patrones_pendientes WHERE confirmaciones < 2 AND fecha < datetime('now', ?)
    """, (f'-{dias} days',)).rowcount
    conn.commit()
    conn.close()
    return eliminados


# ============================================================
#  FUSION DE INTENCIONES
# ============================================================

def fusionar_intenciones(tag_mantener, tag_eliminar):
    """Fusiona tag_eliminar en tag_mantener: mueve patrones y respuestas."""
    conn = conectar()
    row_m = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag_mantener,)).fetchone()
    row_e = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag_eliminar,)).fetchone()
    if not row_m or not row_e:
        conn.close()
        return False

    id_m, id_e = row_m["id"], row_e["id"]
    conn.execute("UPDATE patrones SET intencion_id=? WHERE intencion_id=?", (id_m, id_e))
    conn.execute("UPDATE respuestas SET intencion_id=? WHERE intencion_id=?", (id_m, id_e))
    conn.execute("DELETE FROM intenciones WHERE id=?", (id_e,))
    conn.commit()
    conn.close()
    return True


# ============================================================
#  WRITE BATCHING PARA MENSAJES
# ============================================================

_msg_buffer = []
_msg_buffer_lock = threading.Lock()
_ultimo_flush_msg = _time.time()
MSG_FLUSH_INTERVAL = 10
MSG_FLUSH_SIZE = 100


def guardar_mensaje(sesion, rol, texto, tag_detectado=None, confianza=None):
    with _msg_buffer_lock:
        _msg_buffer.append((sesion, rol, texto, tag_detectado, confianza))
        if len(_msg_buffer) >= MSG_FLUSH_SIZE:
            _flush_mensajes_internal()


def _flush_mensajes_internal():
    global _ultimo_flush_msg
    if not _msg_buffer:
        return
    lote = list(_msg_buffer)
    _msg_buffer.clear()
    _ultimo_flush_msg = _time.time()
    try:
        conn = conectar()
        conn.executemany(
            "INSERT INTO mensajes (sesion,rol,texto,tag_detectado,confianza) VALUES (?,?,?,?,?)", lote)
        conn.execute("UPDATE estadisticas SET valor=valor+? WHERE clave='mensajes_totales'", (len(lote),))
        conn.commit()
        conn.close()
    except Exception:
        pass


def flush_mensajes_periodico():
    with _msg_buffer_lock:
        if _msg_buffer and (_time.time() - _ultimo_flush_msg) >= MSG_FLUSH_INTERVAL:
            _flush_mensajes_internal()


def registrar_sesion_nueva():
    conn = conectar()
    conn.execute("UPDATE estadisticas SET valor=valor+1 WHERE clave='sesiones_totales'")
    conn.commit()
    conn.close()


def incrementar_entrenamientos():
    conn = conectar()
    conn.execute("UPDATE estadisticas SET valor=valor+1 WHERE clave='entrenamientos_totales'")
    conn.commit()
    conn.close()


def crear_intencion_completa(tag, patrones, respuestas):
    conn = conectar()
    total = conn.execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    conn.close()
    if total >= MAX_INTENCIONES:
        return None
    intencion_id = agregar_intencion(tag)
    for patron in patrones:
        agregar_patron(tag, patron, origen="manual", confianza=1.0)
    for respuesta in respuestas:
        agregar_respuesta(tag, respuesta)
    return intencion_id


# ============================================================
#  MANTENIMIENTO
# ============================================================

def decay_confianza_inactivos(dias=30, factor=0.95):
    tags_activos = tags_activos_en_periodo(dias)
    conn = conectar()
    all_tags = [r["tag"] for r in conn.execute("SELECT tag FROM intenciones")]
    afectados = 0
    for tag in all_tags:
        if tag not in tags_activos:
            r = conn.execute("""
                UPDATE patrones SET confianza = confianza * ?
                WHERE intencion_id = (SELECT id FROM intenciones WHERE tag=?)
                AND confianza > 0.05
            """, (factor, tag))
            afectados += r.rowcount
    conn.commit()
    conn.close()
    return afectados


def limpiar_patrones_basura(min_confianza=0.1, dias=60):
    conn = conectar()
    eliminados = conn.execute("""
        DELETE FROM patrones WHERE confianza < ? AND fecha < datetime('now', ?)
    """, (min_confianza, f'-{dias} days')).rowcount
    conn.commit()
    conn.close()
    return eliminados


def desactivar_respuestas_malas(min_peso=0.2):
    conn = conectar()
    desactivadas = conn.execute("""
        UPDATE respuestas SET peso=0.01 WHERE peso < ? AND peso > 0.01
    """, (min_peso,)).rowcount
    conn.commit()
    conn.close()
    return desactivadas


def archivar_intenciones_vacias():
    conn = conectar()
    vacias = conn.execute("""
        DELETE FROM intenciones WHERE id NOT IN (SELECT DISTINCT intencion_id FROM patrones)
    """).rowcount
    conn.commit()
    conn.close()
    return vacias


def purgar_mensajes_viejos(dias=30):
    conn = conectar()
    eliminados = conn.execute(
        "DELETE FROM mensajes WHERE fecha < datetime('now', ?)", (f'-{dias} days',)).rowcount
    conn.execute(
        "DELETE FROM desconocidos WHERE fecha < datetime('now', ?)", (f'-{dias} days',))
    conn.commit()
    conn.close()
    return eliminados


def detectar_patrones_duplicados(umbral=0.9):
    """Detecta patrones casi identicos (Levenshtein > umbral) dentro del mismo tag."""
    from entrenar import similitud_palabras
    conn = conectar()
    duplicados = []
    for row in conn.execute("SELECT id, tag FROM intenciones"):
        pats = conn.execute("SELECT id, texto FROM patrones WHERE intencion_id=? ORDER BY fecha",
                            (row["id"],)).fetchall()
        for i in range(len(pats)):
            for j in range(i + 1, len(pats)):
                if similitud_palabras(pats[i]["texto"].lower(), pats[j]["texto"].lower()) >= umbral:
                    duplicados.append(pats[j]["id"])
    if duplicados:
        placeholders = ",".join("?" * len(duplicados))
        conn.execute(f"DELETE FROM patrones WHERE id IN ({placeholders})", duplicados)
        conn.commit()
    conn.close()
    return len(duplicados)


def renombrar_tags_confusos():
    """Renombra tags de 1-2 chars o solo numeros a 'tema_[id]'."""
    conn = conectar()
    rows = conn.execute("SELECT id, tag FROM intenciones").fetchall()
    renombrados = 0
    for r in rows:
        tag = r["tag"]
        if len(tag) <= 2 or tag.isdigit():
            nuevo = f"tema_{r['id']}"
            try:
                conn.execute("UPDATE intenciones SET tag=? WHERE id=?", (nuevo, r["id"]))
                renombrados += 1
            except sqlite3.IntegrityError:
                pass
    conn.commit()
    conn.close()
    return renombrados


def verificar_limites_ensenanza(tag):
    conn = conectar()
    total_int = conn.execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    row = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
    tp, tr = 0, 0
    if row:
        tp = conn.execute("SELECT COUNT(*) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        tr = conn.execute("SELECT COUNT(*) FROM respuestas WHERE intencion_id=?", (row["id"],)).fetchone()[0]
    conn.close()
    return {
        "intenciones_lleno": total_int >= MAX_INTENCIONES and not row,
        "patrones_lleno": tp >= MAX_PATRONES_POR_INTENCION,
        "respuestas_lleno": tr >= MAX_RESPUESTAS_POR_INTENCION
    }


def vacuum_db():
    """Comprime la base de datos."""
    conn = _crear_conexion_raw()
    conn.execute("VACUUM")
    conn.close()


# ============================================================
#  BACKUPS
# ============================================================

def backup_modelo():
    if os.path.exists(RUTA_MODELO):
        shutil.copy2(RUTA_MODELO, RUTA_BACKUP_MODELO)
        return True
    return False


def restaurar_backup_modelo():
    if os.path.exists(RUTA_BACKUP_MODELO):
        shutil.copy2(RUTA_BACKUP_MODELO, RUTA_MODELO)
        return True
    return False


def backup_db():
    for i in range(3, 1, -1):
        src = os.path.join(RUTA_DATOS, f"ia_backup_{i-1}.db")
        dst = os.path.join(RUTA_DATOS, f"ia_backup_{i}.db")
        if os.path.exists(src):
            shutil.copy2(src, dst)
    dst1 = os.path.join(RUTA_DATOS, "ia_backup_1.db")
    if os.path.exists(RUTA_DB):
        shutil.copy2(RUTA_DB, dst1)


# ============================================================
#  LOG SEMANAL
# ============================================================

def exportar_log_semanal():
    stats = obtener_estadisticas()
    info = obtener_info_salud()
    linea = (f"[{_time.strftime('%Y-%m-%d %H:%M')}] "
             f"Temas: {stats['temas']} | Patrones: {stats['patrones']} | "
             f"Respuestas: {stats['respuestas']} | Mensajes: {stats['mensajes_totales']} | "
             f"DB: {info['db_size_mb']}MB\n")
    try:
        with open(RUTA_LOG, "a") as f:
            f.write(linea)
    except Exception:
        pass
    return linea


# ============================================================
#  SALUD Y MONITOREO
# ============================================================

def obtener_info_salud():
    stats = obtener_estadisticas()
    db_size = os.path.getsize(RUTA_DB) / (1024 * 1024) if os.path.exists(RUTA_DB) else 0
    return {
        "db_size_mb": round(db_size, 2),
        "total_mensajes": stats["mensajes_totales"],
        "total_temas": stats["temas"],
        "total_patrones": stats["patrones"],
        "total_respuestas": stats["respuestas"]
    }
