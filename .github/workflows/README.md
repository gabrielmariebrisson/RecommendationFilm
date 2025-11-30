# GitHub Actions Workflows

Ce dossier contient les workflows GitHub Actions pour l'automatisation CI/CD du projet.

## Workflows Disponibles

### 1. CI (`ci.yml`)

Workflow principal de Continuous Integration qui s'exécute sur chaque push et pull request.

**Jobs inclus :**
- **Test** : Exécute les tests pytest sur Python 3.10, 3.11, 3.12
  - Coverage avec pytest-cov
  - Upload vers Codecov
- **Lint** : Vérification du code avec flake8 et mypy
- **Security** : Scan de sécurité avec Bandit et Safety
- **Build** : Vérification des imports et structure du projet

**Déclencheurs :**
- Push sur `main`, `master`, `develop`
- Pull requests vers `main`, `master`, `develop`

### 2. Performance Benchmarks (`performance.yml`)

Workflow pour exécuter les benchmarks de performance.

**Jobs inclus :**
- **Benchmark** : Exécute les tests de performance
  - Tests à 100, 1k, 10k utilisateurs concurrents
  - Génération du rapport BENCHMARK_RESULTS.md
  - Upload des résultats en artifacts

**Déclencheurs :**
- Manuel (workflow_dispatch)
- Hebdomadaire (chaque lundi à 2h UTC)
- Push sur `main`/`master` si fichiers de performance modifiés

### 3. Release (`release.yml`)

Workflow pour les releases et tags.

**Jobs inclus :**
- **Build and Test** : Tests complets avant release
- **Create Release Notes** : Génération automatique des notes de release

**Déclencheurs :**
- Publication d'une release GitHub
- Manuel avec version spécifiée

## Utilisation

### Exécuter manuellement un workflow

1. Aller dans l'onglet "Actions" du repository
2. Sélectionner le workflow souhaité
3. Cliquer sur "Run workflow"

### Voir les résultats

- Les résultats des tests sont visibles dans l'onglet "Actions"
- Les artifacts (benchmarks, release notes) sont téléchargeables
- Les badges de statut peuvent être ajoutés au README

## Badges (optionnel)

Ajoutez ces badges dans votre README.md :

```markdown
![CI](https://github.com/username/repo/actions/workflows/ci.yml/badge.svg)
![Performance](https://github.com/username/repo/actions/workflows/performance.yml/badge.svg)
```

## Configuration

### Secrets requis

Aucun secret n'est requis pour les workflows de base. Si vous ajoutez :
- Déploiement automatique : configurez les secrets nécessaires
- Notifications : ajoutez les webhooks/slack tokens
- Codecov : le token est automatiquement détecté si le repo est connecté

### Variables d'environnement

Les workflows utilisent les variables par défaut de GitHub Actions. Pour personnaliser :
- Modifiez les fichiers `.github/workflows/*.yml`
- Ajoutez des variables dans Settings > Secrets and variables > Actions

