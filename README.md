# 🧠 Implémentation de FixMatch en PyTorch (MNIST & CIFAR-10)

> **Projet de Deep Learning – Master 2 Data Science (Université Paris-Saclay / Évry)**  
> **Auteurs :** Timoté RICHARD, Henri BELLAMY, Philippe HUYNH, Ziad ABAHAMOU

---

## 🎯 1. Objectif du Projet

L'objectif de ce projet est d'implémenter, d'analyser et d'évaluer de bout en bout l'algorithme **FixMatch** (*Sohn et al., NeurIPS 2020*), une méthode de référence en **apprentissage semi-supervisé (Semi-Supervised Learning - SSL)**.

Dans de nombreux cas d'usage réels, l'acquisition de données brutes est peu coûteuse, mais leur annotation manuelle est longue et onéreuse. Ce projet vise à démontrer comment un réseau de neurones profond peut atteindre des performances élevées en exploitant un **très faible volume de données étiquetées** (ex. : seulement 100 images étiquetées sur les 60 000 de MNIST, ou 250 sur les 50 000 de CIFAR-10) couplé à une grande quantité de données non étiquetées.

---

## ⚙️ 2. Solution Mise en Place

### Principe Algorithmique de FixMatch
FixMatch repose sur la synergie de deux mécanismes clés appliqués à chaque image non étiquetée $u_b$ :

1. **Le Pseudo-Étiquetage (*Pseudo-Labeling*) avec Seuil de Confiance :**
   - L'image non étiquetée subit une **augmentation faible** $\alpha(u_b)$ (padding, recadrage aléatoire).
   - Le modèle prédit une distribution de probabilités $q_b = \text{softmax}(f(\alpha(u_b)))$.
   - Si la confiance maximale dépasse un seuil critique $\tau$ ($\max(q_b) \ge \tau$, typiquement $\tau = 0.95$), la prédiction est convertie en un pseudo-label dur : $\hat{q}_b = \text{argmax}(q_b)$. Dans le cas contraire, l'échantillon est masqué et ignoré pour cette itération.

2. **La Régularisation de Cohérence (*Consistency Regularization*) :**
   - La même image subit parallèlement une **augmentation forte** $\mathcal{A}(u_b)$ via `RandAugment` (perturbations géométriques et photométriques).
   - Le modèle est entraîné à prédire le pseudo-label $\hat{q}_b$ à partir de la version fortement déformée $f(\mathcal{A}(u_b))$.

### Fonction de Perte Globale
L'entraînement minimise la somme pondérée de la perte supervisée $L_s$ et de la perte non supervisée $L_u$ :

$$L_{\text{total}} = L_s + \lambda_u L_u$$

avec :

$$L_u = \frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{I}(\max(q_b) \ge \tau) \, H\left(\hat{q}_b, f(\mathcal{A}(u_b))\right)$$

* $B$ : taille du batch étiqueté (`64`).
* $\mu$ : ratio d'échantillons non étiquetés par batch (`7`, soit `448` images non étiquetées par itération).
* $\tau$ : seuil de confiance (`0.95`).
* $\lambda_u$ : poids de la perte non supervisée (`1.0`).

### Architecture du Modèle & Pipeline Technique
* **Backbone :** `WideResNet-28-2` (~1,47M paramètres), adapté pour accepter à la fois des images en niveaux de gris (1 canal, redimensionnées en $32 \times 32$ par padding pour MNIST) et des images RGB (3 canaux pour CIFAR-10).
* **Dataset Wrapper (`FixMatchDataset`) :** Surcharge de `torch.utils.data.Dataset` permettant de générer simultanément le couple de tenseurs `(vue_faible, vue_forte)` pour chaque image non étiquetée.
* **Optimisation :** Descente de gradient stochastique (`SGD`) avec momentum de Nesterov ($0.9$), *weight decay* ($5 \times 10^{-4}$) et planificateur *Cosine Annealing* (`CosineAnnealingLR`).

---

## 📁 3. Architecture du Projet

```text
fixmatch-pytorch/
│
├── data/                               # Dossier racine des jeux de données (téléchargement automatique)
│   ├── MNIST/                          # Données MNIST brutes et traitées
│   └── cifar-10-batches-py/            # Données CIFAR-10
│
├── notebook/                           # Notebooks Jupyter d'expérimentation et d'analyse
│   ├── Fixmatch_MNIST.ipynb            # Implémentation et résultats complets sur MNIST (100 labels)
│   └── Fixmatch_CIFAR10.ipynb          # Implémentation et résultats complets sur CIFAR-10 (250 labels)
│
├── models/                             # Sauvegardes des poids entraînés (.pth) (optionnel)
│   └── fixmatch_best.pth
│
├── .venv/                              # Environnement virtuel Python (non versionné)
├── requirements.txt                    # Dépendances logicielles du projet
└── README.md                           # Documentation générale du projet
```

---

## 📊 4. Synthèse des Résultats Expérimentaux

Les expériences ont été menées sur **1024 époques** avec un seuil de confiance $\tau = 0.95$ :

| Jeu de données | Labels utilisés | Données non étiquetées | Précision Test (100 ép.) | Précision Test (1024 ép.) | Taux de masque final ($\ge 0.95$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MNIST** | **100** (10 / classe) | 59 900 | 88.77 % | **98.79 %** | ~97 % |
| **CIFAR-10** | **250** (25 / classe) | 49 750 | 47.16 %* | **60.30 %** | ~58 % |

*\*Mesuré à 250 époques sur CIFAR-10.*

### Points Clés Observés
* **Effet d'amorçage (*Bootstrapping*) :** Durant les premières époques, le taux de masque reste proche de $0\%$ car le modèle n'est pas encore assez confiant. Dès que les premières frontières de décision se forment grâce aux labels supervisés, le taux de pseudo-labels retenus croît de manière exponentielle.
* **Robustesse au surapprentissage :** Malgré un volume infime de données étiquetées, la régularisation de cohérence via `RandAugment` empêche le réseau de mémoriser uniquement les exemples annotés et assure une généralisation solide.

---

## 🚀 5. Installation et Exécution

### 1. Cloner le dépôt et activer l'environnement virtuel
```bash
git clone <url-du-depot>
cd fixmatch-pytorch

# Création et activation de l'environnement virtuel (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances
```bash
pip install --upgrade pip
pip install torch torchvision numpy matplotlib seaborn scikit-learn ipykernel
```

### 3. Lancer les notebooks
1. Enregistrer le noyau Jupyter associé à l'environnement :
   ```bash
   python -m ipykernel install --user --name=projet_dl_env --display-name="Python (projet_dl_env)"
   ```
2. Ouvrir `notebook/Fixmatch_MNIST.ipynb` (dans VS Code ou Jupyter Lab), sélectionner le kernel `projet_dl_env` et exécuter les cellules. Les données seront automatiquement téléchargées dans le dossier `../data` à la racine du projet.