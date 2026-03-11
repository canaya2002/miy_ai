"""
Migra datos de datos.json a SQLite.
Se ejecuta automaticamente al iniciar si la base de datos esta vacia.
Asi no se pierde nada de lo que la IA ya aprendio.
"""

import json
import os
import base_datos as bd


def migrar_json_a_sqlite():
    """
    Lee datos.json y los inserta en SQLite.
    Solo migra si la base de datos no tiene intenciones (primera vez).
    """
    # Verificar si ya hay datos en SQLite
    conn = bd.conectar()
    count = conn.execute("SELECT COUNT(*) FROM intenciones").fetchone()[0]
    conn.close()

    if count > 0:
        print("  Base de datos ya tiene datos. Migracion no necesaria.")
        return False

    # Buscar datos.json
    ruta_json = bd.RUTA_JSON
    if not os.path.exists(ruta_json):
        print("  No se encontro datos.json para migrar.")
        return False

    # Leer JSON
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    intenciones = datos.get("intenciones", [])
    if not intenciones:
        print("  datos.json esta vacio.")
        return False

    # Migrar cada intencion
    total_patrones = 0
    total_respuestas = 0

    for intencion in intenciones:
        tag = intencion["tag"]
        patrones = intencion.get("patrones", [])
        respuestas = intencion.get("respuestas", [])

        bd.crear_intencion_completa(tag, patrones, respuestas)
        total_patrones += len(patrones)
        total_respuestas += len(respuestas)

    print(f"  Migrado: {len(intenciones)} temas, {total_patrones} patrones, {total_respuestas} respuestas")
    return True


if __name__ == "__main__":
    bd.crear_tablas()
    migrar_json_a_sqlite()
