"""
Logica de entrenamiento de la red neuronal.
Se usa como modulo (importado por servidor.py) o como script standalone.

Funciones principales:
- limpiar(): limpia texto para procesamiento
- bolsa_de_palabras(): convierte texto a vector numerico
- entrenar_modelo(): entrena la red con datos de SQLite
- cargar_modelo(): carga modelo existente o entrena uno nuevo
"""

import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cerebro import CerebroIA
import base_datos as bd


# ============================================================
#  PROCESAMIENTO DE TEXTO
# ============================================================

def limpiar(texto):
    """Limpia texto: minusculas, sin puntuacion, dividido en palabras"""
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = texto.replace("¿", "").replace("¡", "")
    return texto.split()


def distancia_levenshtein(s1, s2):
    """
    Calcula la distancia de edicion entre dos palabras.
    Cuanto menor el numero, mas parecidas son.
    Ejemplo: "hola" y "hols" tienen distancia 1 (un caracter diferente).
    """
    if len(s1) < len(s2):
        return distancia_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    fila_anterior = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        fila_actual = [i + 1]
        for j, c2 in enumerate(s2):
            costo = 0 if c1 == c2 else 1
            fila_actual.append(min(
                fila_anterior[j + 1] + 1,   # insercion
                fila_actual[j] + 1,          # eliminacion
                fila_anterior[j] + costo     # sustitucion
            ))
        fila_anterior = fila_actual
    return fila_anterior[-1]


def similitud_palabras(p1, p2):
    """Calcula similitud entre 0.0 y 1.0 usando distancia de Levenshtein"""
    dist = distancia_levenshtein(p1, p2)
    max_len = max(len(p1), len(p2))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len


def bolsa_de_palabras(frase, vocabulario, flexible=False):
    """
    Convierte una lista de palabras en un vector numerico (bag of words).
    Cada posicion del vector = 1 si la palabra esta presente, 0 si no.

    Si flexible=True, usa similitud de Levenshtein para tolerar
    errores ortograficos (80%+ de similitud = misma palabra).
    """
    bolsa = [0] * len(vocabulario)
    for palabra in frase:
        for i, v in enumerate(vocabulario):
            if v == palabra:
                bolsa[i] = 1
                break
            elif flexible and len(palabra) > 2 and similitud_palabras(v, palabra) >= 0.8:
                bolsa[i] = 1
                break
    return bolsa


# ============================================================
#  DATASET PARA PYTORCH
# ============================================================

class DatosChat(Dataset):
    """Empaqueta los datos de entrenamiento para PyTorch"""
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


# ============================================================
#  ENTRENAMIENTO
# ============================================================

def entrenar_modelo(epocas=1000, verbose=False):
    """
    Entrena la red neuronal desde cero con todos los datos de SQLite.

    Pasos:
    1. Lee intenciones de la base de datos
    2. Convierte patrones a vectores numericos
    3. Crea y entrena la red neuronal
    4. Guarda el modelo entrenado en disco

    Devuelve: (modelo, vocabulario, tags)
    """
    # Leer datos de SQLite
    intenciones = bd.obtener_intenciones_para_entrenamiento()

    if not intenciones:
        raise ValueError("No hay datos para entrenar. Asegurate de tener datos en la base.")

    # Procesar texto: extraer palabras y tags
    todas_palabras = []
    tags = []
    patrones_tags = []

    for intencion in intenciones:
        tag = intencion["tag"]
        if tag not in tags:
            tags.append(tag)
        for patron in intencion["patrones"]:
            palabras = limpiar(patron)
            todas_palabras.extend(palabras)
            patrones_tags.append((palabras, tag))

    # Vocabulario unico ordenado
    vocabulario = sorted(set(todas_palabras))

    if verbose:
        print(f"  Vocabulario: {len(vocabulario)} palabras")
        print(f"  Intenciones: {len(tags)}")
        print(f"  Patrones: {len(patrones_tags)}")

    # Crear vectores de entrenamiento
    X = []
    Y = []
    for (palabras, tag) in patrones_tags:
        bolsa = bolsa_de_palabras(palabras, vocabulario)
        X.append(bolsa)
        Y.append(tags.index(tag))

    X = np.array(X)
    Y = np.array(Y)

    dataset = DatosChat(X, Y)
    cargador = DataLoader(dataset, batch_size=8, shuffle=True)

    # Crear la red neuronal
    tam_entrada = len(vocabulario)
    tam_salida = len(tags)
    modelo = CerebroIA(tam_entrada, 128, 64, tam_salida)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    # Entrenar
    for epoca in range(epocas):
        error_total = 0
        for (entradas, etiquetas) in cargador:
            prediccion = modelo(entradas)
            error = criterio(prediccion, etiquetas)
            optimizador.zero_grad()
            error.backward()
            optimizador.step()
            error_total += error.item()

        if verbose and (epoca + 1) % 100 == 0:
            print(f"  Epoca {epoca+1}/{epocas} | Error: {error_total:.4f}")

    # Guardar modelo
    estado = {
        "modelo": modelo.state_dict(),
        "vocabulario": vocabulario,
        "tags": tags,
        "dimensiones": {
            "entrada": tam_entrada,
            "oculto_1": 128,
            "oculto_2": 64,
            "salida": tam_salida
        }
    }
    torch.save(estado, bd.RUTA_MODELO)

    if verbose:
        print(f"  Modelo guardado en {bd.RUTA_MODELO}")

    # Registrar en estadisticas
    bd.incrementar_entrenamientos()

    modelo.eval()
    return modelo, vocabulario, tags


def cargar_modelo():
    """Carga el modelo entrenado desde disco. Si no existe, entrena uno nuevo."""
    try:
        estado = torch.load(bd.RUTA_MODELO, weights_only=False)
        vocabulario = estado["vocabulario"]
        tags = estado["tags"]
        dims = estado["dimensiones"]

        modelo = CerebroIA(dims["entrada"], dims["oculto_1"], dims["oculto_2"], dims["salida"])
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        return modelo, vocabulario, tags
    except (FileNotFoundError, Exception) as e:
        print(f"  No se pudo cargar modelo ({e}). Entrenando nuevo...")
        return entrenar_modelo(verbose=True)


# ============================================================
#  EJECUCION COMO SCRIPT
# ============================================================

if __name__ == "__main__":
    bd.crear_tablas()

    # Migrar datos.json si es necesario
    from migrar import migrar_json_a_sqlite
    migrar_json_a_sqlite()

    print("Entrenando modelo...")
    modelo, vocab, tags = entrenar_modelo(verbose=True)
    print(f"\nListo! {len(tags)} intenciones, {len(vocab)} palabras en vocabulario.")
