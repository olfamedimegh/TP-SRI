# 🦙 Guide d'Installation d'Ollama sur Windows

Ce guide explique comment configurer Ollama pour utiliser l'évaluation "LLM as a Judge" du TP2 sur un PC Windows.

---

## 📋 Configuration Requise

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| **Système** | Windows 10 | Windows 11 |
| **RAM** | 8 GB | 16 GB |
| **Espace disque** | 10 GB | 20 GB |
| **Processeur** | 4 cœurs | 8 cœurs |

---

## 🚀 Installation

### Étape 1 : Télécharger Ollama

1. Ouvrez votre navigateur et allez sur : **https://ollama.ai/download**
2. Cliquez sur le bouton **"Download for Windows"**
3. Attendez que le fichier `OllamaSetup.exe` soit téléchargé

### Étape 2 : Installer Ollama

1. Double-cliquez sur `OllamaSetup.exe`
2. Suivez les instructions de l'installateur
3. Une fois installé, Ollama démarre automatiquement en arrière-plan
4. Vous verrez une icône 🦙 dans la barre des tâches (en bas à droite)

### Étape 3 : Télécharger un Modèle de Langage

Ouvrez **PowerShell** ou **Invite de commandes** (CMD) et exécutez :

```powershell
# Option 1 : Mistral (recommandé - bon équilibre qualité/vitesse)
ollama pull mistral

# Option 2 : Llama 2 (modèle par défaut)
ollama pull llama2

# Option 3 : Gemma 2B (léger, pour PC avec moins de RAM)
ollama pull gemma:2b
```

⏳ **Note** : Le téléchargement peut prendre plusieurs minutes selon votre connexion internet.

---

## 🐍 Configuration Python

### Installer la dépendance requise

```powershell
pip install requests
```

---

## ▶️ Utilisation avec le TP2

### Lancer l'évaluation

```powershell
# Naviguer vers le dossier du TP
cd C:\chemin\vers\TP2

# Lancer l'évaluation avec Ollama
python main.py --mode evaluate --provider ollama --num-queries 5
```

### Exemples de commandes

```powershell
# Mode démonstration (sans LLM)
python main.py --mode demo

# Mode interactif
python main.py --mode interactive

# Évaluation complète avec Ollama
python main.py --mode evaluate --provider ollama

# Évaluation avec un nombre limité de requêtes
python main.py --mode evaluate --provider ollama --num-queries 3
```

---

## 🔧 Dépannage

### Problème : "Ollama ne répond pas"

**Solution 1** : Vérifier que Ollama est en cours d'exécution
- Regardez si l'icône 🦙 est présente dans la barre des tâches
- Si non, recherchez "Ollama" dans le menu Démarrer et lancez-le

**Solution 2** : Redémarrer le service
- Clic droit sur l'icône Ollama dans la barre des tâches
- Cliquez sur "Quit" puis relancez Ollama

**Solution 3** : Lancer manuellement
```powershell
ollama serve
```

### Problème : "Modèle non trouvé"

Vérifiez les modèles installés :
```powershell
ollama list
```

Si le modèle n'apparaît pas, téléchargez-le :
```powershell
ollama pull mistral
```

### Problème : "Mémoire insuffisante"

Utilisez un modèle plus léger :
```powershell
ollama pull gemma:2b
```

Ou fermez les applications gourmandes en mémoire avant de lancer l'évaluation.

### Problème : "Connection refused"

Le serveur Ollama n'est pas démarré. Lancez :
```powershell
ollama serve
```

---

## 📊 Modèles Disponibles

| Modèle | Taille | RAM Requise | Qualité | Vitesse |
|--------|--------|-------------|---------|---------|
| `gemma:2b` | 1.4 GB | 4 GB | ⭐⭐ | ⚡⚡⚡ |
| `llama2` | 3.8 GB | 8 GB | ⭐⭐⭐ | ⚡⚡ |
| `mistral` | 4.1 GB | 8 GB | ⭐⭐⭐⭐ | ⚡⚡ |
| `llama2:13b` | 7.3 GB | 16 GB | ⭐⭐⭐⭐⭐ | ⚡ |

**Recommandation** : Utilisez `mistral` pour un bon compromis entre qualité et performance.

---

## ✅ Vérification de l'Installation

Testez que tout fonctionne :

```powershell
# 1. Vérifier qu'Ollama répond
curl http://localhost:11434/api/tags

# 2. Lister les modèles installés
ollama list

# 3. Tester un modèle
ollama run mistral "Dis bonjour en français"

# 4. Tester avec le TP (mode MOCK d'abord)
python main.py --mode evaluate --provider mock --num-queries 2

# 5. Tester avec Ollama
python main.py --mode evaluate --provider ollama --num-queries 2
```

---

## 💡 Conseils

1. **Fermez les applications lourdes** avant de lancer l'évaluation (navigateurs avec beaucoup d'onglets, etc.)

2. **Utilisez un SSD** si possible - les modèles se chargent plus rapidement

3. **Première exécution lente** - Le modèle est chargé en mémoire lors de la première requête, les suivantes seront plus rapides

4. **Mode économique** - Si votre PC est limité, utilisez `--num-queries 3` pour réduire le nombre d'évaluations

---

## 🆘 Support

En cas de problème :
1. Consultez la documentation officielle : https://ollama.ai
2. Vérifiez les issues GitHub : https://github.com/ollama/ollama

---

*INSAT - GL5 - Systèmes de Recherche d'Information - 2025*

