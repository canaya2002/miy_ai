"""
Logica de entrenamiento de la red neuronal.
Incluye: early stopping, epocas adaptativas, backup y validacion.
"""

import string
import random
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
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = texto.replace("¿", "").replace("¡", "")
    return texto.split()


def distancia_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return distancia_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    fila_anterior = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        fila_actual = [i + 1]
        for j, c2 in enumerate(s2):
            costo = 0 if c1 == c2 else 1
            fila_actual.append(min(fila_anterior[j+1]+1, fila_actual[j]+1, fila_anterior[j]+costo))
        fila_anterior = fila_actual
    return fila_anterior[-1]


def similitud_palabras(p1, p2):
    dist = distancia_levenshtein(p1, p2)
    ml = max(len(p1), len(p2))
    return 1.0 - dist / ml if ml > 0 else 1.0


def bolsa_de_palabras(frase, vocabulario, flexible=False):
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


class DatosChat(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return len(self.X)


# ============================================================
#  ENTRENAMIENTO CON EARLY STOPPING Y EPOCAS ADAPTATIVAS
# ============================================================

def calcular_epocas(num_patrones):
    """Menos datos = mas epocas. Mas datos = menos epocas (pero early stopping)."""
    if num_patrones < 100:
        return 1000, 0       # pocas: 1000 epocas, sin early stopping
    elif num_patrones < 500:
        return 500, 50        # medias: 500 max, early stop si no mejora en 50
    else:
        return 300, 50        # muchas: 300 max, early stop en 50


def entrenar_modelo(epocas=None, verbose=False):
    """
    Entrena con:
    - Backup del modelo anterior
    - Epocas adaptativas segun cantidad de datos
    - Early stopping con validacion (10% holdout)
    - Validacion post-entrenamiento (si es peor, revierte al backup)
    """
    # Backup antes de entrenar
    bd.backup_modelo()

    intenciones = bd.obtener_intenciones_para_entrenamiento()
    if not intenciones:
        raise ValueError("No hay datos para entrenar")

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

    vocabulario = sorted(set(todas_palabras))

    if verbose:
        print(f"  Vocabulario: {len(vocabulario)} | Tags: {len(tags)} | Patrones: {len(patrones_tags)}")

    X = []
    Y = []
    for (palabras, tag) in patrones_tags:
        bolsa = bolsa_de_palabras(palabras, vocabulario)
        X.append(bolsa)
        Y.append(tags.index(tag))

    X = np.array(X)
    Y = np.array(Y)

    # Dividir en entrenamiento (90%) y validacion (10%)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = max(1, len(indices) // 10)
    val_idx = indices[:split]
    train_idx = indices[split:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    dataset = DatosChat(X_train, Y_train)
    cargador = DataLoader(dataset, batch_size=8, shuffle=True)

    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.LongTensor(Y_val)

    # Epocas adaptativas
    max_epocas, paciencia = calcular_epocas(len(patrones_tags))
    if epocas is not None:
        max_epocas = epocas
        paciencia = 0

    tam_entrada = len(vocabulario)
    tam_salida = len(tags)
    modelo = CerebroIA(tam_entrada, 128, 64, tam_salida)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    mejor_val_loss = float('inf')
    epocas_sin_mejora = 0
    mejor_estado = None

    for epoca in range(max_epocas):
        modelo.train()
        error_total = 0
        for (entradas, etiquetas) in cargador:
            prediccion = modelo(entradas)
            error = criterio(prediccion, etiquetas)
            optimizador.zero_grad()
            error.backward()
            optimizador.step()
            error_total += error.item()

        # Validacion
        if paciencia > 0 and len(X_val) > 0:
            modelo.eval()
            with torch.no_grad():
                val_pred = modelo(X_val_t)
                val_loss = criterio(val_pred, Y_val_t).item()

            if val_loss < mejor_val_loss:
                mejor_val_loss = val_loss
                epocas_sin_mejora = 0
                mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            else:
                epocas_sin_mejora += 1

            if epocas_sin_mejora >= paciencia:
                if verbose:
                    print(f"  Early stopping en epoca {epoca+1} (sin mejora en {paciencia} epocas)")
                if mejor_estado:
                    modelo.load_state_dict(mejor_estado)
                break

        if verbose and (epoca + 1) % 100 == 0:
            extra = f" | Val: {val_loss:.4f}" if paciencia > 0 and len(X_val) > 0 else ""
            print(f"  Epoca {epoca+1}/{max_epocas} | Error: {error_total:.4f}{extra}")

    # Restaurar mejor estado si usamos early stopping
    if mejor_estado and paciencia > 0:
        modelo.load_state_dict(mejor_estado)

    modelo.eval()

    # Validar accuracy en TODO el dataset
    X_all_t = torch.FloatTensor(X)
    Y_all_t = torch.LongTensor(Y)
    with torch.no_grad():
        pred_all = modelo(X_all_t)
        _, predicted = torch.max(pred_all, 1)
        accuracy = (predicted == Y_all_t).sum().item() / len(Y_all_t)

    if verbose:
        print(f"  Accuracy: {accuracy:.1%}")

    # Si accuracy es muy baja y hay backup, revertir
    if accuracy < 0.5 and os.path.exists(bd.RUTA_BACKUP_MODELO):
        if verbose:
            print(f"  Accuracy muy baja ({accuracy:.1%}). Revirtiendo al backup.")
        bd.restaurar_backup_modelo()
        return cargar_modelo()

    # Guardar modelo
    estado = {
        "modelo": modelo.state_dict(),
        "vocabulario": vocabulario,
        "tags": tags,
        "dimensiones": {"entrada": tam_entrada, "oculto_1": 128, "oculto_2": 64, "salida": tam_salida}
    }
    torch.save(estado, bd.RUTA_MODELO)

    bd.incrementar_entrenamientos()

    if verbose:
        print(f"  Modelo guardado. Accuracy: {accuracy:.1%}")

    return modelo, vocabulario, tags


def cargar_modelo():
    try:
        estado = torch.load(bd.RUTA_MODELO, weights_only=False)
        vocabulario = estado["vocabulario"]
        tags = estado["tags"]
        dims = estado["dimensiones"]
        modelo = CerebroIA(dims["entrada"], dims["oculto_1"], dims["oculto_2"], dims["salida"])
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        return modelo, vocabulario, tags
    except Exception as e:
        print(f"  No se pudo cargar modelo ({e}). Entrenando nuevo...")
        return entrenar_modelo(verbose=True)


if __name__ == "__main__":
    bd.crear_tablas()
    from migrar import migrar_json_a_sqlite
    migrar_json_a_sqlite()
    print("Entrenando modelo...")
    modelo, vocab, tags = entrenar_modelo(verbose=True)
    print(f"\nListo! {len(tags)} intenciones, {len(vocab)} palabras.")
