#!/bin/bash
# ============================================================
#  Setup completo para Oracle Cloud Always Free (ARM, Ubuntu)
#  4 OCPUs, 24 GB RAM, $0/mes
# ============================================================
set -e

echo "=== Actualizando sistema ==="
sudo apt update && sudo apt upgrade -y

echo "=== Instalando dependencias ==="
sudo apt install -y python3 python3-pip python3-venv nginx git

echo "=== Clonando repositorio ==="
cd /home/ubuntu
if [ ! -d "miy_ai" ]; then
    git clone https://github.com/TU_USUARIO/miy_ai.git
fi
cd miy_ai

echo "=== Creando entorno virtual ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Instalando dependencias Python ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Creando directorio de datos ==="
sudo mkdir -p /data
sudo chown ubuntu:ubuntu /data

echo "=== Configurando Nginx ==="
sudo cp nginx.conf /etc/nginx/sites-available/ia
sudo ln -sf /etc/nginx/sites-available/ia /etc/nginx/sites-enabled/ia
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

echo "=== Configurando servicios systemd ==="
sudo cp ia-web.service /etc/systemd/system/
sudo cp ia-trainer.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "=== Iniciando servicios ==="
sudo systemctl enable ia-web ia-trainer
sudo systemctl start ia-web
sudo systemctl start ia-trainer

echo "=== Abriendo puertos en firewall ==="
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
# Persistir reglas
sudo apt install -y iptables-persistent
sudo netfilter-persistent save

echo ""
echo "============================================"
echo "  Setup completado!"
echo "  Web:     http://$(curl -s ifconfig.me)"
echo "  Health:  http://$(curl -s ifconfig.me)/health"
echo ""
echo "  Comandos utiles:"
echo "    sudo systemctl status ia-web"
echo "    sudo systemctl status ia-trainer"
echo "    sudo journalctl -u ia-web -f"
echo "    sudo journalctl -u ia-trainer -f"
echo "============================================"
