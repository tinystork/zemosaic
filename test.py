# diagnostic_zemosaic.py
import sys
import os
import traceback

print("=== DIAGNOSTIC ZEMOSAIC ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Script location: {__file__ if '__file__' in globals() else 'Interactive'}")
print()

# Test 1: Import direct de reproject
print("1. Test import direct de reproject:")
try:
    import reproject
    print(f"   ✓ reproject importé avec succès - Version: {getattr(reproject, '__version__', 'inconnue')}")
    print(f"   ✓ Localisation: {reproject.__file__}")
except Exception as e:
    print(f"   ❌ Erreur import reproject: {e}")
    traceback.print_exc()
print()

# Test 2: Import des sous-modules de reproject
print("2. Test import sous-modules reproject:")
for module_name in ['reproject.mosaicking', 'reproject.spherical_intersect']:
    try:
        module = __import__(module_name, fromlist=[''])
        print(f"   ✓ {module_name} importé avec succès")
        print(f"     Localisation: {getattr(module, '__file__', 'pas de __file__')}")
    except Exception as e:
        print(f"   ❌ Erreur import {module_name}: {e}")

print()

# Test 3: Vérifier si zemosaic_worker.py existe et son contenu
print("3. Vérification zemosaic_worker.py:")
worker_file = "zemosaic_worker.py"
if os.path.exists(worker_file):
    print(f"   ✓ {worker_file} existe")
    try:
        with open(worker_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            print(f"   ✓ Taille: {len(content)} caractères, {len(lines)} lignes")
            
            # Chercher les imports de reproject
            import_lines = [i for i, line in enumerate(lines, 1) 
                          if 'import' in line and 'reproject' in line]
            
            if import_lines:
                print("   📋 Lignes contenant 'import' et 'reproject':")
                for line_num in import_lines:
                    print(f"      Ligne {line_num}: {lines[line_num-1].strip()}")
            else:
                print("   ⚠️  Aucune ligne d'import reproject trouvée")
                
    except Exception as e:
        print(f"   ❌ Erreur lecture {worker_file}: {e}")
else:
    print(f"   ❌ {worker_file} n'existe pas dans le répertoire courant")
    print("   📁 Fichiers .py présents:")
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    for py_file in py_files:
        print(f"      {py_file}")

print()

# Test 4: Tentative d'import de zemosaic_worker
print("4. Test import zemosaic_worker:")
try:
    import zemosaic_worker
    print("   ✓ zemosaic_worker importé avec succès")
    
    # Vérifier si la fonction principale existe
    if hasattr(zemosaic_worker, 'run_hierarchical_mosaic'):
        print("   ✓ Fonction run_hierarchical_mosaic trouvée")
    else:
        print("   ⚠️  Fonction run_hierarchical_mosaic non trouvée")
        print("   📋 Attributs disponibles:")
        attrs = [attr for attr in dir(zemosaic_worker) if not attr.startswith('_')]
        for attr in attrs[:10]:  # Limiter l'affichage
            print(f"      {attr}")
        if len(attrs) > 10:
            print(f"      ... et {len(attrs)-10} autres")
            
except Exception as e:
    print(f"   ❌ Erreur import zemosaic_worker: {e}")
    traceback.print_exc()

print()

# Test 5: Variables d'environnement Python
print("5. Variables d'environnement Python:")
python_path = os.environ.get('PYTHONPATH', '')
if python_path:
    print(f"   PYTHONPATH: {python_path}")
else:
    print("   PYTHONPATH: (non défini)")

print()
print("=== FIN DIAGNOSTIC ===")