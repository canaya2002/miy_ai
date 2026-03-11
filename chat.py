import json
import string
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from cerebro import CerebroIA

# ============================================================
#  FUNCIONES BASE
# ============================================================

def limpiar(texto):
    texto = texto.lower()
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = texto.replace("¿", "").replace("¡", "")
    return texto.split()

def bolsa_de_palabras(frase, vocabulario):
    bolsa = [0] * len(vocabulario)
    for palabra in frase:
        for i, v in enumerate(vocabulario):
            if v == palabra:
                bolsa[i] = 1
    return bolsa

# ============================================================
#  ENTRENAR (se usa al inicio y cada vez que aprende algo nuevo)
# ============================================================

def entrenar_modelo():
    """Entrena la red neuronal desde cero con los datos actuales"""
    with open("datos.json", "r", encoding="utf-8") as f:
        datos = json.load(f)

    todas_palabras = []
    tags = []
    patrones_tags = []

    for intencion in datos["intenciones"]:
        tag = intencion["tag"]
        if tag not in tags:
            tags.append(tag)
        for patron in intencion["patrones"]:
            palabras = limpiar(patron)
            todas_palabras.extend(palabras)
            patrones_tags.append((palabras, tag))

    vocabulario = sorted(set(todas_palabras))

    X = []
    Y = []
    for (palabras, tag) in patrones_tags:
        bolsa = bolsa_de_palabras(palabras, vocabulario)
        X.append(bolsa)
        Y.append(tags.index(tag))

    X = np.array(X)
    Y = np.array(Y)

    class DatosChat(Dataset):
        def __init__(self, X, Y):
            self.X = torch.FloatTensor(X)
            self.Y = torch.LongTensor(Y)
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
        def __len__(self):
            return len(self.X)

    dataset = DatosChat(X, Y)
    cargador = DataLoader(dataset, batch_size=8, shuffle=True)

    tam_entrada = len(vocabulario)
    tam_salida = len(tags)

    modelo = CerebroIA(tam_entrada, 128, 64, tam_salida)
    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

    for epoca in range(1000):
        for (entradas, etiquetas) in cargador:
            prediccion = modelo(entradas)
            error = criterio(prediccion, etiquetas)
            optimizador.zero_grad()
            error.backward()
            optimizador.step()

    estado = {
        "modelo": modelo.state_dict(),
        "vocabulario": vocabulario,
        "tags": tags,
        "datos": datos,
        "dimensiones": {
            "entrada": tam_entrada,
            "oculto_1": 128,
            "oculto_2": 64,
            "salida": tam_salida
        }
    }
    torch.save(estado, "ia_entrenada.pth")
    return modelo, vocabulario, tags, datos

# ============================================================
#  CARGAR IA (si ya existe entrenada, la carga; si no, entrena)
# ============================================================

def cargar_modelo():
    try:
        estado = torch.load("ia_entrenada.pth", weights_only=False)
        vocabulario = estado["vocabulario"]
        tags = estado["tags"]
        datos = estado["datos"]
        dims = estado["dimensiones"]

        modelo = CerebroIA(dims["entrada"], dims["oculto_1"], dims["oculto_2"], dims["salida"])
        modelo.load_state_dict(estado["modelo"])
        modelo.eval()
        return modelo, vocabulario, tags, datos
    except FileNotFoundError:
        print("No hay IA entrenada. Entrenando por primera vez...")
        return entrenar_modelo()

# ============================================================
#  APRENDER ALGO NUEVO
# ============================================================

def aprender(datos, frase_usuario):
    """Cuando la IA no entiende, le pregunta al usuario y aprende"""

    print("IA: No entendi eso. Quieres ensenarme?")
    respuesta = input("    (si/no): ").strip().lower()

    if respuesta not in ["si", "s", "si"]:
        print("IA: Ok, no hay problema.")
        return datos, False

    # Preguntar si es una intencion nueva o existente
    print("\nIA: Es un tema que ya conozco o es algo nuevo?")
    print("    Temas que conozco:")
    for i, intencion in enumerate(datos["intenciones"]):
        print(f"      {i + 1}. {intencion['tag']}")
    print(f"      0. Es un tema NUEVO")

    opcion = input("    Numero: ").strip()

    if opcion == "0":
        # Tema nuevo
        tag = input("IA: Como se llama este tema? (una palabra, ej: musica): ").strip().lower()
        resp = input("IA: Y que te deberia responder cuando me hablen de eso?: ").strip()

        nueva_intencion = {
            "tag": tag,
            "patrones": [frase_usuario],
            "respuestas": [resp]
        }
        datos["intenciones"].append(nueva_intencion)

    else:
        # Agregar a tema existente
        try:
            idx = int(opcion) - 1
            intencion = datos["intenciones"][idx]
            intencion["patrones"].append(frase_usuario)

            print(f"IA: Listo, ahora se que '{frase_usuario}' es sobre '{intencion['tag']}'.")
            agregar_resp = input("IA: Quieres agregarle una respuesta nueva tambien? (si/no): ").strip().lower()
            if agregar_resp in ["si", "s", "si"]:
                resp = input("IA: Cual?: ").strip()
                intencion["respuestas"].append(resp)
        except (ValueError, IndexError):
            print("IA: No entendi la opcion. No aprendi nada esta vez.")
            return datos, False

    # Guardar datos actualizados
    with open("datos.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    return datos, True

# ============================================================
#  RESPONDER
# ============================================================

def obtener_respuesta(texto, modelo, vocabulario, tags, datos):
    palabras = limpiar(texto)
    bolsa = bolsa_de_palabras(palabras, vocabulario)
    tensor = torch.FloatTensor(bolsa).unsqueeze(0)

    with torch.no_grad():
        resultado = modelo(tensor)

    probabilidades = torch.softmax(resultado, dim=1)
    confianza, prediccion = torch.max(probabilidades, dim=1)

    tag = tags[prediccion.item()]
    confianza = confianza.item()

    if confianza > 0.6:
        for intencion in datos["intenciones"]:
            if intencion["tag"] == tag:
                respuesta = random.choice(intencion["respuestas"])
                return respuesta, tag, confianza, True

    return None, "???", confianza, False

# ============================================================
#  CHAT PRINCIPAL
# ============================================================

print("Cargando IA...")
modelo, vocabulario, tags, datos = cargar_modelo()

print("\n" + "=" * 50)
print("  Tu IA esta lista (aprende mientras hablas)")
print("  Comandos:")
print("    'salir'  - Terminar")
print("    'debug'  - Ver como piensa")
print("    'stats'  - Ver cuanto sabe")
print("=" * 50)

modo_debug = False

while True:
    entrada = input("\nTu: ").strip()

    if not entrada:
        continue

    if entrada.lower() == "salir":
        print("IA: Hasta luego! Hoy aprendi cosas nuevas.")
        break

    if entrada.lower() == "debug":
        modo_debug = not modo_debug
        print(f"(Debug {'activado' if modo_debug else 'desactivado'})")
        continue

    if entrada.lower() == "stats":
        print(f"\n--- Estadisticas de tu IA ---")
        print(f"  Temas que conoce: {len(datos['intenciones'])}")
        total_patrones = sum(len(i['patrones']) for i in datos['intenciones'])
        total_respuestas = sum(len(i['respuestas']) for i in datos['intenciones'])
        print(f"  Patrones aprendidos: {total_patrones}")
        print(f"  Respuestas posibles: {total_respuestas}")
        print(f"  Vocabulario: {len(vocabulario)} palabras")
        print(f"-----------------------------")
        continue

    respuesta, tag, confianza, entendio = obtener_respuesta(entrada, modelo, vocabulario, tags, datos)

    if modo_debug:
        print(f"  [debug] Intencion: {tag} | Confianza: {confianza:.1%}")

    if entendio:
        print(f"IA: {respuesta}")
    else:
        # No entendio - oportunidad de aprender
        datos, aprendio = aprender(datos, entrada)

        if aprendio:
            print("\nIA: Aprendido! Re-entrenando mi cerebro...")
            modelo, vocabulario, tags, datos = entrenar_modelo()
            modelo.eval()
            print("IA: Listo! Ya se sobre eso. Pruebame.")
