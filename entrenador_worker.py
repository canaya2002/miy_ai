"""
Worker dedicado de entrenamiento.
Corre como proceso separado. Vigila signal files y tabla staging.
Ejecuta mantenimiento diario.
"""

import os
import time
import logging
import sys
import schedule

import base_datos as bd
import entrenar
import auto_aprendizaje as auto

# ============================================================
#  CONFIGURACION
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WORKER] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_RETRAIN = os.path.join(_BASE_DIR, "NEEDS_RETRAIN")
SIGNAL_UPDATED = os.path.join(_BASE_DIR, "MODEL_UPDATED")

MIN_INTERVALO_RETRAIN = 120  # minimo 2 min entre reentrenamientos
_ultimo_retrain = 0


# ============================================================
#  FLUSH PATRONES PENDIENTES (staging → patrones reales)
# ============================================================

def flush_patrones_pendientes():
    """Mueve patrones de la tabla staging a la tabla real."""
    conn = bd.conectar()
    pendientes = conn.execute("SELECT id, tag, texto, origen, confianza FROM patrones_pendientes").fetchall()
    if not pendientes:
        conn.close()
        return 0

    procesados = 0
    ids_borrar = []
    for p in pendientes:
        ids_borrar.append(p["id"])
        bd.agregar_patron(p["tag"], p["texto"], origen=p["origen"], confianza=p["confianza"])
        procesados += 1

    if ids_borrar:
        placeholders = ",".join("?" * len(ids_borrar))
        conn.execute(f"DELETE FROM patrones_pendientes WHERE id IN ({placeholders})", ids_borrar)
        conn.commit()

    conn.close()
    if procesados:
        logger.info(f"  Flush staging: {procesados} patrones procesados")
    return procesados


# ============================================================
#  REENTRENAMIENTO
# ============================================================

def verificar_y_reentrenar():
    """Verifica si hay signal de reentrenamiento y ejecuta."""
    global _ultimo_retrain

    ahora = time.time()
    if ahora - _ultimo_retrain < MIN_INTERVALO_RETRAIN:
        return

    if not os.path.exists(SIGNAL_RETRAIN):
        return

    try:
        os.remove(SIGNAL_RETRAIN)
    except Exception:
        pass

    logger.info("Signal de reentrenamiento detectada.")

    # Flush staging primero
    flush_patrones_pendientes()

    # Reentrenar
    try:
        logger.info("Entrenando modelo...")
        entrenar.entrenar_modelo(verbose=True)
        _ultimo_retrain = time.time()

        # Signal al servidor para hot-reload
        with open(SIGNAL_UPDATED, "w") as f:
            f.write(str(time.time()))
        logger.info("Entrenamiento completado. Signal MODEL_UPDATED escrita.")
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")


# ============================================================
#  MANTENIMIENTO DIARIO
# ============================================================

def mantenimiento_diario():
    """Ejecuta todas las tareas de mantenimiento."""
    logger.info("Iniciando mantenimiento diario...")
    flush_patrones_pendientes()
    resultado = auto.ejecutar_mantenimiento()
    logger.info(f"Mantenimiento completado: {resultado}")

    # Reentrenar despues del mantenimiento
    try:
        entrenar.entrenar_modelo(verbose=True)
        with open(SIGNAL_UPDATED, "w") as f:
            f.write(str(time.time()))
        logger.info("Reentrenamiento post-mantenimiento OK.")
    except Exception as e:
        logger.error(f"Error reentrenamiento post-mantenimiento: {e}")


# ============================================================
#  LOOP PRINCIPAL
# ============================================================

def main():
    logger.info("========================================")
    logger.info("  Worker de entrenamiento iniciado")
    logger.info("========================================")

    bd.crear_tablas()

    # Programar mantenimiento diario a las 4:00 AM
    schedule.every().day.at("04:00").do(mantenimiento_diario)

    logger.info("Vigilando signals cada 30s. Mantenimiento programado a las 04:00.")

    while True:
        try:
            verificar_y_reentrenar()
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Error en loop principal: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
