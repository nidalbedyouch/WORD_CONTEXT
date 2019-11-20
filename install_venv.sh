echo "suppression du dossier venv..."
rm -rf venv
echo "création de l'environnement virtuel ..."
virtualenv -p python3 venv
echo "installation des packages python ..."
source venv/bin/activate
pip3 install -r requirements.txt
deactivate
echo "fin d'installation de l'environnement virtuel ..."
