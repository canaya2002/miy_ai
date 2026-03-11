"""
Worker dedicado de entrenamiento.
Corre como proceso separado. Vigila signal files y tabla staging.
Ejecuta mantenimiento diario, VACUUM semanal, auto-ajuste de hiperparametros.
"""

import os
import time
import logging
import sys
import schedule
import datetime

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

MIN_INTERVALO_RETRAIN = 120
_ultimo_retrain = 0
_patrones_nuevos_desde_ultimo = 0
UMBRAL_REENTRENAMIENTO_COMPLETO = 50


# ============================================================
#  FLUSH PATRONES PENDIENTES (staging -> patrones reales)
# ============================================================

def flush_patrones_pendientes():
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
#  REENTRENAMIENTO (con soporte incremental)
# ============================================================

def verificar_y_reentrenar():
    global _ultimo_retrain, _patrones_nuevos_desde_ultimo

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

    # Flush staging y promover confirmados
    procesados = flush_patrones_pendientes()
    bd.promover_patrones_confirmados(min_confirmaciones=2)
    _patrones_nuevos_desde_ultimo += procesados

    # Decidir: incremental o completo
    try:
        if _patrones_nuevos_desde_ultimo < UMBRAL_REENTRENAMIENTO_COMPLETO:
            logger.info("Intentando entrenamiento incremental...")
            resultado = entrenar.entrenar_incremental(verbose=True)
            if resultado:
                _ultimo_retrain = time.time()
                _patrones_nuevos_desde_ultimo = 0
                _signal_updated()
                return

        logger.info("Entrenamiento completo...")
        entrenar.entrenar_modelo(verbose=True)
        _ultimo_retrain = time.time()
        _patrones_nuevos_desde_ultimo = 0
        _signal_updated()

    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")


def _signal_updated():
    try:
        with open(SIGNAL_UPDATED, "w") as f:
            f.write(str(time.time()))
        logger.info("Entrenamiento completado. Signal MODEL_UPDATED escrita.")
    except Exception:
        pass


# ============================================================
#  MANTENIMIENTO DIARIO (4:00 AM)
# ============================================================

def mantenimiento_diario():
    logger.info("Iniciando mantenimiento diario...")
    flush_patrones_pendientes()
    resultado = auto.ejecutar_mantenimiento()
    logger.info(f"Mantenimiento completado: {resultado}")

    # Reentrenar despues del mantenimiento
    try:
        entrenar.entrenar_modelo(verbose=True)
        _signal_updated()
        logger.info("Reentrenamiento post-mantenimiento OK.")
    except Exception as e:
        logger.error(f"Error reentrenamiento post-mantenimiento: {e}")


# ============================================================
#  AUTO-AJUSTE DE HIPERPARAMETROS (diario 5:00 AM)
# ============================================================

def ajuste_hiperparametros():
    logger.info("Iniciando auto-ajuste de hiperparametros...")
    try:
        resultado = entrenar.auto_ajustar_hiperparametros(verbose=True)
        if resultado:
            logger.info(f"Mejores hiperparametros: {resultado}")
        else:
            logger.info("No se encontraron mejores hiperparametros (datos insuficientes).")
    except Exception as e:
        logger.error(f"Error en auto-ajuste: {e}")


# ============================================================
#  VACUUM SEMANAL (domingos 3:00 AM)
# ============================================================

def vacuum_semanal():
    logger.info("Ejecutando VACUUM semanal...")
    try:
        bd.vacuum_db()
        logger.info("VACUUM completado.")
    except Exception as e:
        logger.error(f"Error en VACUUM: {e}")


# ============================================================
#  LOG SEMANAL
# ============================================================

def log_semanal():
    logger.info("Exportando log semanal...")
    try:
        linea = bd.exportar_log_semanal()
        logger.info(f"Log: {linea.strip()}")
    except Exception as e:
        logger.error(f"Error exportando log: {e}")


# ============================================================
#  LOOP PRINCIPAL
# ============================================================

def main():
    logger.info("========================================")
    logger.info("  Worker de entrenamiento iniciado")
    logger.info("========================================")

    bd.crear_tablas()

    # Programar tareas
    schedule.every().day.at("04:00").do(mantenimiento_diario)
    schedule.every().day.at("05:00").do(ajuste_hiperparametros)
    schedule.every().sunday.at("03:00").do(vacuum_semanal)
    schedule.every().sunday.at("03:30").do(log_semanal)

    logger.info("Vigilando signals cada 30s.")
    logger.info("Mantenimiento: 04:00 | Hiperparametros: 05:00 | VACUUM: dom 03:00")

    while True:
        try:
            verificar_y_reentrenar()
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Error en loop principal: {e}")
        time.sleep(30)


if __name__ == "__main__":
    main()
