# Installiamo i pacchetti python richiesti, la utility gdown verrà installata da pip se necessario
pip3 install -r requirements.txt

# Scarichiamo il dataset da Google Drive
gdown 1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo

# Decomprimiamo il dataset compresso scaricato dentro la cartella dataset/
unzip emovo.zip -d dataset

# Rimuoviamo il dataset compresso per non lasciare file temporanei
rm emovo.zip
