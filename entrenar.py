"""
Logica de entrenamiento de la red neuronal.
Incluye: early stopping, epocas adaptativas, backup, validacion,
entrenamiento incremental, auto-ajuste de hiperparametros.
"""

import string
import random
import os
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
    texto = texto.replace("\u00bf", "").replace("\u00a1", "")
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


def aplicar_sinonimos(palabras, mapa_sinonimos):
    """Reemplaza palabras por su forma canonica segun el diccionario de sinonimos."""
    if not mapa_sinonimos:
        return palabras
    return [mapa_sinonimos.get(p, p) for p in palabras]


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
#  PREPARAR DATOS
# ============================================================

def _preparar_datos(intenciones, mapa_sinonimos=None):
    """Prepara vocabulario, X, Y a partir de intenciones."""
    todas_palabras = []
    tags = []
    patrones_tags = []

    for intencion in intenciones:
        tag = intencion["tag"]
        if tag not in tags:
            tags.append(tag)
        for patron in intencion["patrones"]:
            palabras = limpiar(patron)
            palabras = aplicar_sinonimos(palabras, mapa_sinonimos)
            todas_palabras.extend(palabras)
            patrones_tags.append((palabras, tag))

    vocabulario = sorted(set(todas_palabras))

    X, Y = [], []
    for (palabras, tag) in patrones_tags:
        bolsa = bolsa_de_palabras(palabras, vocabulario)
        X.append(bolsa)
        Y.append(tags.index(tag))

    return np.array(X), np.array(Y), vocabulario, tags, len(patrones_tags)


# ============================================================
#  ENTRENAMIENTO CON EARLY STOPPING Y EPOCAS ADAPTATIVAS
# ============================================================

def calcular_epocas(num_patrones):
    if num_patrones < 100:
        return 1000, 0
    elif num_patrones < 500:
        return 500, 50
    else:
        return 300, 50


def entrenar_modelo(epocas=None, verbose=False, lr=None, batch_size=None):
    """
    Entrena con backup, epocas adaptativas, early stopping, validacion.
    Usa mejores hiperparametros si estan disponibles.
    """
    bd.backup_modelo()

    intenciones = bd.obtener_intenciones_para_entrenamiento()
    if not intenciones:
        raise ValueError("No hay datos para entrenar")

    # Cargar sinonimos
    mapa_sinonimos = bd.obtener_sinonimos()

    # Cargar mejores hiperparametros si no se pasan explicitamente
    if lr is None or batch_size is None:
        mejores = bd.obtener_mejores_hiperparametros()
        if mejores:
            if lr is None:
                lr = mejores["lr"]
            if batch_size is None:
                batch_size = mejores["batch_size"]

    if lr is None:
        lr = 0.001
    if batch_size is None:
        batch_size = 8

    X, Y, vocabulario, tags, num_patrones = _preparar_datos(intenciones, mapa_sinonimos)

    if verbose:
        print(f"  Vocabulario: {len(vocabulario)} | Tags: {len(tags)} | Patrones: {num_patrones}")

    # Split train/val
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = max(1, len(indices) // 10)
    val_idx = indices[:split]
    train_idx = indices[split:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    dataset = DatosChat(X_train, Y_train)
    cargador = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.LongTensor(Y_val)

    max_epocas, paciencia = calcular_epocas(num_patrones)
    if epocas is not None:
        max_epocas = epocas
        paciencia = 0

    tam_entrada = len(vocabulario)
    tam_salida = len(tags)
    modelo = CerebroIA(tam_entrada, 128, 64, tam_salida)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

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

    if mejor_estado and paciencia > 0:
        modelo.load_state_dict(mejor_estado)

    modelo.eval()

    # Accuracy en todo el dataset
    X_all_t = torch.FloatTensor(X)
    Y_all_t = torch.LongTensor(Y)
    with torch.no_grad():
        pred_all = modelo(X_all_t)
        _, predicted = torch.max(pred_all, 1)
        accuracy = (predicted == Y_all_t).sum().item() / len(Y_all_t)

    if verbose:
        print(f"  Accuracy: {accuracy:.1%}")

    if accuracy < 0.5 and os.path.exists(bd.RUTA_BACKUP_MODELO):
        if verbose:
            print(f"  Accuracy muy baja ({accuracy:.1%}). Revirtiendo al backup.")
        bd.restaurar_backup_modelo()
        return cargar_modelo()

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


# ============================================================
#  ENTRENAMIENTO INCREMENTAL
# ============================================================

def entrenar_incremental(verbose=False):
    """
    Fine-tuning: carga modelo existente y entrena solo con datos recientes
    usando un learning rate bajo. Mucho mas rapido que reentrenamiento completo.
    Retorna (modelo, vocab, tags) o None si falla.
    """
    if not os.path.exists(bd.RUTA_MODELO):
        return None

    try:
        estado_prev = torch.load(bd.RUTA_MODELO, weights_only=False)
        vocab_prev = estado_prev["vocabulario"]
        tags_prev = estado_prev["tags"]
        dims = estado_prev["dimensiones"]
    except Exception:
        return None

    intenciones = bd.obtener_intenciones_para_entrenamiento()
    if not intenciones:
        return None

    mapa_sinonimos = bd.obtener_sinonimos()
    X, Y, vocabulario, tags, num_patrones = _preparar_datos(intenciones, mapa_sinonimos)

    # Si el vocabulario o los tags cambiaron, no podemos hacer incremental
    if vocabulario != vocab_prev or tags != tags_prev:
        if verbose:
            print("  Incremental: vocabulario/tags cambiaron, se requiere reentrenamiento completo.")
        return None

    # Cargar modelo previo
    modelo = CerebroIA(dims["entrada"], dims["oculto_1"], dims["oculto_2"], dims["salida"])
    modelo.load_state_dict(estado_prev["modelo"])

    # Accuracy antes
    X_all_t = torch.FloatTensor(X)
    Y_all_t = torch.LongTensor(Y)
    modelo.eval()
    with torch.no_grad():
        pred = modelo(X_all_t)
        _, predicted = torch.max(pred, 1)
        accuracy_antes = (predicted == Y_all_t).sum().item() / len(Y_all_t)

    # Fine-tune con LR bajo
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.0001)
    dataset = DatosChat(X, Y)
    cargador = DataLoader(dataset, batch_size=8, shuffle=True)

    modelo.train()
    for epoca in range(150):
        for (entradas, etiquetas) in cargador:
            prediccion = modelo(entradas)
            error = criterio(prediccion, etiquetas)
            optimizador.zero_grad()
            error.backward()
            optimizador.step()

    # Accuracy despues
    modelo.eval()
    with torch.no_grad():
        pred = modelo(X_all_t)
        _, predicted = torch.max(pred, 1)
        accuracy_despues = (predicted == Y_all_t).sum().item() / len(Y_all_t)

    if verbose:
        print(f"  Incremental: accuracy {accuracy_antes:.1%} -> {accuracy_despues:.1%}")

    # Si empeoro mas del 5%, revertir
    if accuracy_despues < accuracy_antes - 0.05:
        if verbose:
            print("  Incremental: accuracy empeoro, revirtiendo.")
        return None

    # Guardar
    estado = {
        "modelo": modelo.state_dict(),
        "vocabulario": vocabulario,
        "tags": tags,
        "dimensiones": dims
    }
    torch.save(estado, bd.RUTA_MODELO)

    if verbose:
        print(f"  Incremental: modelo actualizado. Accuracy: {accuracy_despues:.1%}")

    return modelo, vocabulario, tags


# ============================================================
#  AUTO-AJUSTE DE HIPERPARAMETROS
# ============================================================

def auto_ajustar_hiperparametros(verbose=False):
    """
    Prueba combinaciones de hiperparametros y guarda la mejor.
    Usa 50% de datos para entrenar y 50% para evaluar.
    """
    intenciones = bd.obtener_intenciones_para_entrenamiento()
    if not intenciones or len(intenciones) < 3:
        return None

    mapa_sinonimos = bd.obtener_sinonimos()
    X, Y, vocabulario, tags, _ = _preparar_datos(intenciones, mapa_sinonimos)

    if len(X) < 10:
        return None

    # Split 50/50
    indices = list(range(len(X)))
    random.shuffle(indices)
    mid = len(indices) // 2
    train_idx = indices[:mid]
    test_idx = indices[mid:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    X_test_t = torch.FloatTensor(X_test)
    Y_test_t = torch.LongTensor(Y_test)

    tam_entrada = len(vocabulario)
    tam_salida = len(tags)

    lrs = [0.0001, 0.0005, 0.001, 0.005]
    epocas_list = [300, 500, 800]
    batch_sizes = [8, 16, 32]

    mejor_accuracy = 0
    mejor_config = None

    for lr_val in lrs:
        for ep_val in epocas_list:
            for bs_val in batch_sizes:
                try:
                    modelo = CerebroIA(tam_entrada, 128, 64, tam_salida)
                    criterio = nn.CrossEntropyLoss()
                    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr_val)
                    dataset = DatosChat(X_train, Y_train)
                    cargador = DataLoader(dataset, batch_size=bs_val, shuffle=True)

                    modelo.train()
                    for _ in range(min(ep_val, 200)):  # cap para que no tarde tanto
                        for (entradas, etiquetas) in cargador:
                            prediccion = modelo(entradas)
                            error = criterio(prediccion, etiquetas)
                            optimizador.zero_grad()
                            error.backward()
                            optimizador.step()

                    modelo.eval()
                    with torch.no_grad():
                        pred = modelo(X_test_t)
                        _, predicted = torch.max(pred, 1)
                        accuracy = (predicted == Y_test_t).sum().item() / len(Y_test_t)

                    if accuracy > mejor_accuracy:
                        mejor_accuracy = accuracy
                        mejor_config = {"lr": lr_val, "epocas": ep_val, "batch_size": bs_val}

                except Exception:
                    continue

    if mejor_config:
        bd.guardar_hiperparametros(
            mejor_config["lr"], mejor_config["epocas"],
            mejor_config["batch_size"], mejor_accuracy)
        if verbose:
            print(f"  Mejor config: lr={mejor_config['lr']}, "
                  f"epocas={mejor_config['epocas']}, "
                  f"batch={mejor_config['batch_size']}, "
                  f"accuracy={mejor_accuracy:.1%}")

    return mejor_config


# ============================================================
#  CARGAR MODELO
# ============================================================

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
