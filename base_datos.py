"""
Base de datos SQLite para la IA.
Reemplaza datos.json para manejar muchos datos y muchos usuarios simultáneos.

Tablas:
- intenciones: los temas que la IA conoce
- patrones: las formas en que la gente dice algo (asociadas a una intencion)
- respuestas: lo que la IA puede contestar (con peso para feedback)
- mensajes: historial completo de conversaciones
- estadisticas: contadores globales
"""

import sqlite3
import os
import threading

# ============================================================
#  RUTAS DE DATOS
#  En Render: /data/ (disco persistente que sobrevive redeployments)
#  En local: la carpeta del proyecto
# ============================================================

def obtener_ruta_datos():
    """Detecta si estamos en Render (con disco persistente) o en local"""
    if os.path.exists("/data"):
        return "/data"
    return os.path.dirname(os.path.abspath(__file__))

RUTA_DATOS = obtener_ruta_datos()
RUTA_DB = os.path.join(RUTA_DATOS, "ia.db")
RUTA_MODELO = os.path.join(RUTA_DATOS, "ia_entrenada.pth")
RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos.json")

# Lock global para escrituras concurrentes
_lock_db = threading.Lock()


def conectar():
    """Crea una conexion a SQLite con WAL para lecturas concurrentes"""
    conn = sqlite3.connect(RUTA_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def crear_tablas():
    """Crea todas las tablas si no existen"""
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
#  OPERACIONES DE LECTURA
# ============================================================

def obtener_intenciones_para_entrenamiento():
    """
    Devuelve todas las intenciones con sus patrones y respuestas,
    en formato listo para el entrenador.
    """
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
            intenciones.append({
                "tag": row["tag"],
                "patrones": patrones,
                "respuestas": respuestas
            })

    conn.close()
    return intenciones


def obtener_respuestas_por_tag(tag):
    """Devuelve las respuestas de un tag con sus pesos e IDs"""
    conn = conectar()
    rows = conn.execute("""
        SELECT r.id, r.texto, r.peso FROM respuestas r
        JOIN intenciones i ON r.intencion_id = i.id
        WHERE i.tag = ?
    """, (tag,)).fetchall()
    conn.close()
    return [{"id": r["id"], "texto": r["texto"], "peso": r["peso"]} for r in rows]


def obtener_estadisticas():
    """Devuelve estadisticas completas de la IA"""
    conn = conectar()

    temas = conn.execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    patrones = conn.execute("SELECT COUNT(*) FROM patrones").fetchone()[0]
    respuestas = conn.execute("SELECT COUNT(*) FROM respuestas").fetchone()[0]

    mensajes_totales = conn.execute(
        "SELECT valor FROM estadisticas WHERE clave = 'mensajes_totales'"
    ).fetchone()[0]
    sesiones_totales = conn.execute(
        "SELECT valor FROM estadisticas WHERE clave = 'sesiones_totales'"
    ).fetchone()[0]
    entrenamientos = conn.execute(
        "SELECT valor FROM estadisticas WHERE clave = 'entrenamientos_totales'"
    ).fetchone()[0]

    lista_temas = [r["tag"] for r in conn.execute("SELECT tag FROM intenciones ORDER BY tag")]

    conn.close()

    return {
        "temas": temas,
        "patrones": patrones,
        "respuestas": respuestas,
        "mensajes_totales": mensajes_totales,
        "sesiones_totales": sesiones_totales,
        "entrenamientos": entrenamientos,
        "lista_temas": lista_temas
    }


def obtener_temas_detallados():
    """Devuelve info detallada de cada tema (para la pagina /cerebro)"""
    conn = conectar()
    temas = []

    for row in conn.execute("SELECT id, tag FROM intenciones ORDER BY tag"):
        num_patrones = conn.execute(
            "SELECT COUNT(*) FROM patrones WHERE intencion_id = ?", (row["id"],)
        ).fetchone()[0]
        num_respuestas = conn.execute(
            "SELECT COUNT(*) FROM respuestas WHERE intencion_id = ?", (row["id"],)
        ).fetchone()[0]
        primera_fecha = conn.execute(
            "SELECT MIN(fecha) FROM patrones WHERE intencion_id = ?", (row["id"],)
        ).fetchone()[0]

        # Contar origenes de los patrones (manual, auto_refuerzo, auto_contexto)
        origenes = {}
        for o in conn.execute(
            "SELECT origen, COUNT(*) as c FROM patrones WHERE intencion_id = ? GROUP BY origen",
            (row["id"],)
        ):
            origenes[o["origen"]] = o["c"]

        temas.append({
            "tag": row["tag"],
            "patrones": num_patrones,
            "respuestas": num_respuestas,
            "fecha_creacion": primera_fecha,
            "origenes": origenes
        })

    conn.close()
    return temas


# ============================================================
#  OPERACIONES DE ESCRITURA
# ============================================================

def agregar_intencion(tag):
    """Agrega un tag nuevo. Devuelve su ID (existente o nuevo)."""
    with _lock_db:
        conn = conectar()
        try:
            conn.execute("INSERT INTO intenciones (tag) VALUES (?)", (tag,))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Ya existe, no pasa nada
        id_tag = conn.execute("SELECT id FROM intenciones WHERE tag = ?", (tag,)).fetchone()[0]
        conn.close()
        return id_tag


def agregar_patron(tag, texto, origen="manual", confianza=1.0):
    """Agrega un patron a una intencion. Devuelve True si se agrego."""
    with _lock_db:
        conn = conectar()
        row = conn.execute("SELECT id FROM intenciones WHERE tag = ?", (tag,)).fetchone()
        if not row:
            conn.close()
            return False

        # No duplicar patrones
        existe = conn.execute(
            "SELECT id FROM patrones WHERE intencion_id = ? AND texto = ?",
            (row["id"], texto)
        ).fetchone()

        if not existe:
            conn.execute(
                "INSERT INTO patrones (intencion_id, texto, origen, confianza) VALUES (?, ?, ?, ?)",
                (row["id"], texto, origen, confianza)
            )
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False


def agregar_respuesta(tag, texto, peso=1.0):
    """Agrega una respuesta a una intencion. Devuelve True si se agrego."""
    with _lock_db:
        conn = conectar()
        row = conn.execute("SELECT id FROM intenciones WHERE tag = ?", (tag,)).fetchone()
        if not row:
            conn.close()
            return False

        existe = conn.execute(
            "SELECT id FROM respuestas WHERE intencion_id = ? AND texto = ?",
            (row["id"], texto)
        ).fetchone()

        if not existe:
            conn.execute(
                "INSERT INTO respuestas (intencion_id, texto, peso) VALUES (?, ?, ?)",
                (row["id"], texto, peso)
            )
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False


def ajustar_peso_respuesta(respuesta_id, delta):
    """Sube o baja el peso de una respuesta (minimo 0.1)"""
    with _lock_db:
        conn = conectar()
        conn.execute(
            "UPDATE respuestas SET peso = MAX(0.1, peso + ?) WHERE id = ?",
            (delta, respuesta_id)
        )
        conn.commit()
        conn.close()


def guardar_mensaje(sesion, rol, texto, tag_detectado=None, confianza=None):
    """Guarda un mensaje en el historial y actualiza el contador"""
    with _lock_db:
        conn = conectar()
        conn.execute(
            "INSERT INTO mensajes (sesion, rol, texto, tag_detectado, confianza) VALUES (?, ?, ?, ?, ?)",
            (sesion, rol, texto, tag_detectado, confianza)
        )
        conn.execute(
            "UPDATE estadisticas SET valor = valor + 1 WHERE clave = 'mensajes_totales'"
        )
        conn.commit()
        conn.close()


def registrar_sesion_nueva():
    """Incrementa el contador de sesiones unicas"""
    with _lock_db:
        conn = conectar()
        conn.execute(
            "UPDATE estadisticas SET valor = valor + 1 WHERE clave = 'sesiones_totales'"
        )
        conn.commit()
        conn.close()


def incrementar_entrenamientos():
    """Incrementa el contador de re-entrenamientos"""
    with _lock_db:
        conn = conectar()
        conn.execute(
            "UPDATE estadisticas SET valor = valor + 1 WHERE clave = 'entrenamientos_totales'"
        )
        conn.commit()
        conn.close()


def crear_intencion_completa(tag, patrones, respuestas):
    """Crea una intencion con todos sus patrones y respuestas de una vez"""
    intencion_id = agregar_intencion(tag)
    for patron in patrones:
        agregar_patron(tag, patron, origen="manual", confianza=1.0)
    for respuesta in respuestas:
        agregar_respuesta(tag, respuesta)
    return intencion_id
