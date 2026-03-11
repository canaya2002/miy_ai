import torch
import torch.nn as nn


class CerebroIA(nn.Module):
    """
    Red neuronal desde cero para clasificar intenciones.
    
    Arquitectura:
    - Capa de entrada: recibe el texto convertido en numeros
    - Capa oculta 1: 128 neuronas (aprenden patrones basicos)
    - Capa oculta 2: 64 neuronas (aprenden patrones complejos)
    - Capa de salida: una neurona por cada intencion posible
    
    La IA aprende a clasificar lo que le dices en una "intencion"
    y luego elige una respuesta de esa categoria.
    """

    def __init__(self, tam_entrada, tam_oculto_1, tam_oculto_2, tam_salida):
        super(CerebroIA, self).__init__()

        # Estas son las capas de neuronas
        self.capa1 = nn.Linear(tam_entrada, tam_oculto_1)
        self.capa2 = nn.Linear(tam_oculto_1, tam_oculto_2)
        self.capa3 = nn.Linear(tam_oculto_2, tam_salida)

        # Funciones de activacion (hacen que la red pueda aprender patrones no lineales)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Evita que memorice en vez de aprender

    def forward(self, x):
        # Asi fluye la informacion por la red:
        x = self.capa1(x)       # Entrada -> Capa 1
        x = self.relu(x)        # Activacion
        x = self.dropout(x)     # Anti-memorizacion
        x = self.capa2(x)       # Capa 1 -> Capa 2
        x = self.relu(x)        # Activacion
        x = self.dropout(x)     # Anti-memorizacion
        x = self.capa3(x)       # Capa 2 -> Salida
        return x
