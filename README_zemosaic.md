ZeMosaic - README

English

Overview

ZeMosaic is a hierarchical mosaicker software tailored for astrophotography images, especially suited for managing large datasets captured by Seestar telescopes.

Installation

Clone the repository:

git clone https://github.com/yourusername/ZeMosaic.git
cd ZeMosaic

Set up Python environment:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

Dependencies:
Ensure you have installed:

Python 3.10+

Astropy

Reproject

Astroalign

OpenCV

These dependencies are listed in the requirements.txt file.

Running ZeMosaic

To launch ZeMosaic, execute:

python run_zemosaic.py

A graphical user interface will open, enabling you to select your input and output directories, and configure various mosaicking parameters.

Français

Présentation

ZeMosaic est un logiciel de mosaïquage hiérarchique conçu pour les images d'astrophotographie, particulièrement adapté à la gestion de grands ensembles d'images capturées par des télescopes Seestar.

Installation

Cloner le dépôt :

git clone https://github.com/yourusername/ZeMosaic.git
cd ZeMosaic

Configurer l'environnement Python :

python -m venv venv
source venv/bin/activate  # Sur Windows utilisez : venv\Scripts\activate
pip install -r requirements.txt

lancer Zemosaic : 
python run_zemosaic.py

Dépendances :
Assurez-vous d'avoir installé :

Python 3.10+

Astropy

Reproject

Astroalign

OpenCV

Ces dépendances sont listées dans le fichier requirements.txt.

Lancer ZeMosaic

Pour démarrer ZeMosaic, exécutez :

python run_zemosaic.py

Une interface graphique s'ouvrira, vous permettant de sélectionner vos dossiers d'entrée et de sortie, et de configurer divers paramètres de mosaïquage.