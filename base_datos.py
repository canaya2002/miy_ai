"""
Base de datos SQLite para la IA.
Maneja intenciones, patrones, respuestas, mensajes, estadisticas.
Incluye funciones de limpieza, decay, limites, y salud.
"""

import sqlite3
import os
import re
import threading
import shutil

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

_lock_db = threading.Lock()

# ============================================================
#  LIMITES DE SEGURIDAD
# ============================================================

MAX_INTENCIONES = 1000
MAX_PATRONES_POR_INTENCION = 100
MAX_RESPUESTAS_POR_INTENCION = 50
MAX_DB_SIZE_MB = 500

# ============================================================
#  CONEXION
# ============================================================

def conectar():
    conn = sqlite3.connect(RUTA_DB, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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
        INSERT OR IGNORE INTO estadisticas (clave, valor) VALUES ('mensajes_totales', 0);
        INSERT OR IGNORE INTO estadisticas (clave, valor) VALUES ('sesiones_totales', 0);
        INSERT OR IGNORE INTO estadisticas (clave, valor) VALUES ('entrenamientos_totales', 0);
    """)
    conn.commit()
    conn.close()


# ============================================================
#  LECTURA
# ============================================================

def obtener_intenciones_para_entrenamiento():
    conn = conectar()
    intenciones = []
    for row in conn.execute("SELECT id, tag FROM intenciones ORDER BY id"):
        patrones = [p["texto"] for p in conn.execute(
            "SELECT texto FROM patrones WHERE intencion_id = ?", (row["id"],)
        )]
        respuestas = [{"texto": r["texto"], "peso": r["peso"], "id": r["id"]} for r in conn.execute(
            "SELECT id, texto, peso FROM respuestas WHERE intencion_id = ?", (row["id"],)
        )]
        if patrones and respuestas:
            intenciones.append({"tag": row["tag"], "patrones": patrones, "respuestas": respuestas})
    conn.close()
    return intenciones


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
        np = conn.execute("SELECT COUNT(*) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        nr = conn.execute("SELECT COUNT(*) FROM respuestas WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        fecha = conn.execute("SELECT MIN(fecha) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        origenes = {}
        for o in conn.execute("SELECT origen, COUNT(*) as c FROM patrones WHERE intencion_id=? GROUP BY origen", (row["id"],)):
            origenes[o["origen"]] = o["c"]
        temas.append({"tag": row["tag"], "patrones": np, "respuestas": nr, "fecha_creacion": fecha, "origenes": origenes})
    conn.close()
    return temas


def obtener_palabras_por_tag():
    """Devuelve {tag: set(palabras)} para calcular conexiones en el cerebro"""
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


# ============================================================
#  FRASES NO ENTENDIDAS (para clustering)
# ============================================================

def obtener_frases_no_entendidas(dias=7):
    """Devuelve frases que la IA no entendio, con info de sesion"""
    conn = conectar()
    rows = conn.execute("""
        SELECT texto, sesion FROM mensajes
        WHERE rol='usuario' AND (confianza IS NULL OR confianza < 0.6)
        AND fecha > datetime('now', ?)
    """, (f'-{dias} days',)).fetchall()
    conn.close()
    return [{"texto": r["texto"], "sesion": r["sesion"]} for r in rows]


# ============================================================
#  TAGS ACTIVOS (para decay)
# ============================================================

def tags_activos_en_periodo(dias=30):
    conn = conectar()
    rows = conn.execute("""
        SELECT DISTINCT tag_detectado FROM mensajes
        WHERE tag_detectado IS NOT NULL AND confianza > 0.6
        AND fecha > datetime('now', ?)
    """, (f'-{dias} days',)).fetchall()
    conn.close()
    return {r[0] for r in rows}


# ============================================================
#  ESCRITURA
# ============================================================

def agregar_intencion(tag):
    with _lock_db:
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
    with _lock_db:
        conn = conectar()
        row = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
        if not row:
            conn.close()
            return False
        existe = conn.execute("SELECT id FROM patrones WHERE intencion_id=? AND texto=?", (row["id"], texto)).fetchone()
        if not existe:
            # Verificar limite
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
    with _lock_db:
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


def ajustar_peso_respuesta(respuesta_id, delta):
    with _lock_db:
        conn = conectar()
        conn.execute("UPDATE respuestas SET peso=MAX(0.1, peso+?) WHERE id=?", (delta, respuesta_id))
        conn.commit()
        conn.close()


def guardar_mensaje(sesion, rol, texto, tag_detectado=None, confianza=None):
    with _lock_db:
        conn = conectar()
        conn.execute("INSERT INTO mensajes (sesion,rol,texto,tag_detectado,confianza) VALUES (?,?,?,?,?)",
                     (sesion, rol, texto, tag_detectado, confianza))
        conn.execute("UPDATE estadisticas SET valor=valor+1 WHERE clave='mensajes_totales'")
        conn.commit()
        conn.close()


def registrar_sesion_nueva():
    with _lock_db:
        conn = conectar()
        conn.execute("UPDATE estadisticas SET valor=valor+1 WHERE clave='sesiones_totales'")
        conn.commit()
        conn.close()


def incrementar_entrenamientos():
    with _lock_db:
        conn = conectar()
        conn.execute("UPDATE estadisticas SET valor=valor+1 WHERE clave='entrenamientos_totales'")
        conn.commit()
        conn.close()


def crear_intencion_completa(tag, patrones, respuestas):
    total = conectar().execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    if total >= MAX_INTENCIONES:
        return None
    intencion_id = agregar_intencion(tag)
    for patron in patrones:
        agregar_patron(tag, patron, origen="manual", confianza=1.0)
    for respuesta in respuestas:
        agregar_respuesta(tag, respuesta)
    return intencion_id


# ============================================================
#  MANTENIMIENTO (decay, limpieza, backups)
# ============================================================

def decay_confianza_inactivos(dias=30, factor=0.95):
    """Reduce confianza de patrones de tags que nadie ha usado en N dias"""
    tags_activos = tags_activos_en_periodo(dias)
    with _lock_db:
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
    """Elimina patrones con confianza muy baja y viejos"""
    with _lock_db:
        conn = conectar()
        eliminados = conn.execute("""
            DELETE FROM patrones WHERE confianza < ? AND fecha < datetime('now', ?)
        """, (min_confianza, f'-{dias} days')).rowcount
        conn.commit()
        conn.close()
    return eliminados


def desactivar_respuestas_malas(min_peso=0.2):
    """Desactiva respuestas con peso muy bajo"""
    with _lock_db:
        conn = conectar()
        desactivadas = conn.execute("""
            UPDATE respuestas SET peso=0.01 WHERE peso < ? AND peso > 0.01
        """, (min_peso,)).rowcount
        conn.commit()
        conn.close()
    return desactivadas


def archivar_intenciones_vacias():
    """Elimina intenciones sin patrones activos"""
    with _lock_db:
        conn = conectar()
        vacias = conn.execute("""
            DELETE FROM intenciones WHERE id NOT IN (SELECT DISTINCT intencion_id FROM patrones)
        """).rowcount
        conn.commit()
        conn.close()
    return vacias


def verificar_limites_ensenanza(tag):
    """Verifica si se pueden agregar mas datos a un tag"""
    conn = conectar()
    total_int = conn.execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    row = conn.execute("SELECT id FROM intenciones WHERE tag=?", (tag,)).fetchone()
    tp = 0
    tr = 0
    if row:
        tp = conn.execute("SELECT COUNT(*) FROM patrones WHERE intencion_id=?", (row["id"],)).fetchone()[0]
        tr = conn.execute("SELECT COUNT(*) FROM respuestas WHERE intencion_id=?", (row["id"],)).fetchone()[0]
    conn.close()
    return {
        "intenciones_lleno": total_int >= MAX_INTENCIONES and not row,
        "patrones_lleno": tp >= MAX_PATRONES_POR_INTENCION,
        "respuestas_lleno": tr >= MAX_RESPUESTAS_POR_INTENCION
    }


# ============================================================
#  BACKUPS
# ============================================================

def backup_modelo():
    """Copia el modelo actual a backup antes de re-entrenar"""
    if os.path.exists(RUTA_MODELO):
        shutil.copy2(RUTA_MODELO, RUTA_BACKUP_MODELO)
        return True
    return False


def restaurar_backup_modelo():
    """Restaura el modelo desde el backup"""
    if os.path.exists(RUTA_BACKUP_MODELO):
        shutil.copy2(RUTA_BACKUP_MODELO, RUTA_MODELO)
        return True
    return False


def backup_db():
    """Copia la base de datos (rotacion: max 3 backups)"""
    for i in range(3, 1, -1):
        src = os.path.join(RUTA_DATOS, f"ia_backup_{i-1}.db")
        dst = os.path.join(RUTA_DATOS, f"ia_backup_{i}.db")
        if os.path.exists(src):
            shutil.copy2(src, dst)
    dst1 = os.path.join(RUTA_DATOS, "ia_backup_1.db")
    if os.path.exists(RUTA_DB):
        shutil.copy2(RUTA_DB, dst1)


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
