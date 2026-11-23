# Règles de génération des hypothèses (Agent Theorist)

Ce document décrit le fonctionnement détaillé de la génération d'hypothèses dans **HypoLab**, aussi bien lorsqu'un **LLM est disponible** que lorsqu'**aucun LLM n'est disponible** (mode heuristique déterministe).

---

## 1. Point d'entrée et détection du LLM

La génération d'hypothèses est exécutée à l'étape 2 du pipeline d'orchestration (`run_scientific_investigation` dans `hypolab/agents/orchestrator.py`), via la fonction `generate_hypotheses()` dans `hypolab/agents/theorist.py`, à partir du `DatasetProfile` produit par le profileur (`hypolab/profiler/engine.py`).

### Critères de disponibilité d'un LLM

Le mode LLM est tenté si `use_llm=True` (désactivable notamment via `--no-llm` en CLI) **et** si `is_llm_configured(config)` (`hypolab/agents/llm_client.py`) retourne `True`.

Un LLM est considéré configuré dans l'un des cas suivants (après chargement du fichier `.env`) :
1. Le nom du modèle commence par `ollama/` ou `ollama_chat/` (modèle local Ollama).
2. Une clé API explicite valide est fournie dans `config.api_key`.
3. Au moins une des variables d'environnement suivantes est définie avec une valeur non factice (ne commençant pas par `your_` et ne se terminant pas par `_here`) :
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GROQ_API_KEY`
   - `MISTRAL_API_KEY`

---

## 2. Cas 1 : Un LLM est disponible

### 2.1 Construction des prompts

- **Prompt système (`THEORIST_SYSTEM_PROMPT`)** : demande au modèle d'agir comme un chercheur quantitatif (*Theorist Agent*) et de renvoyer **uniquement** un objet JSON contenant la clé `"hypotheses"` (liste d'objets conformes au schéma `Hypothesis` de `hypolab/models.py`). Chaque hypothèse doit contenir :
  - `id` : identifiant (`"HYP-001"`, `"HYP-002"`, …) ;
  - `title` : titre court ;
  - `statement` : hypothèse alternative ($H_1$) falsifiable ;
  - `null_hypothesis` : hypothèse nulle ($H_0$) ;
  - `rationale` : justification statistique fondée sur le profil ;
  - `variables` : noms exacts des colonnes principales impliquées ;
  - `confounders` : 0 à 2 noms de colonnes confondantes potentielles (`[]` si aucune) ;
  - `proposed_method` : une valeur parmi :
    - `compare_two_groups`
    - `compare_multiple_groups`
    - `test_categorical_association`
    - `test_correlation`
    - `multivariate_regression`
    - `python_sandbox`
  - `tool_params` : dictionnaire des paramètres attendus par la méthode choisie :
    - `compare_two_groups` : `{"group_col": "<cat_col>", "target_col": "<num_col>"}`
    - `compare_multiple_groups` : `{"group_col": "<cat_col>", "target_col": "<num_col>"}`
    - `test_categorical_association` : `{"col_a": "<cat_col_1>", "col_b": "<cat_col_2>"}`
    - `test_correlation` : `{"col_x": "<num_col_1>", "col_y": "<num_col_2>", "control_col": "<optional_confounder>"}`
    - `multivariate_regression` : `{"target": "<target_col>", "features": ["<col1>", "<col2>"]}`
    - `python_sandbox` : `{"code": "<code python assignant result (dict) et fig (plotly)>"}`
  - `priority` : entier de `1` (plus haute) à `5` (plus basse).

- **Prompt utilisateur** : fournit :
  - l'objectif de recherche (`research_goal`) ;
  - le plafond d'hypothèses (`config.max_hypotheses`, `6` par défaut) ;
  - le résumé compact du jeu de données généré par `DatasetProfile.to_llm_summary()` : dimensions, nombre de doublons, profil de chaque colonne (type sémantique, distribution, valeurs uniques, valeurs manquantes, statistiques descriptives ou top 5 catégories), jusqu'aux 10 meilleures corrélations numériques (Pearson $r$ / Spearman $\rho$) et alertes de qualité.

### 2.2 Appel, tolérance aux pannes et extraction JSON

- L'appel est effectué via `call_llm_json()` (`litellm.completion`) avec `temperature=0.2` par défaut (`hypolab/config.py`).
- Pour les modèles `gemini/*`, en cas d'erreur `404`, `503`, `no longer available` ou `UNAVAILABLE`, une bascule automatique est tentée sur une liste de modèles de secours (`gemini/gemini-3.8-flash`, `gemini/gemini-2.5-flash-lite`, `gemini/gemini-3.1-flash-lite`).
- La réponse texte est parsée par `extract_json_from_text()` et `_loads_lenient()`, qui :
  - extraient le contenu des blocs Markdown ```` ```json ... ``` ```` ou des bornes `{...}` / `[...]` ;
  - réparent les séquences d'échappement invalides (ex. notations LaTeX `\alpha`, `\beta`, `\le`) et les virgules finales.

### 2.3 Normalisation, validation et repli

Pour chaque élément reçu (tronqué à `config.max_hypotheses`) :
1. `id` est réassigné séquentiellement (`HYP-001`, `HYP-002`, …).
2. `confounders` est filtré pour ne conserver que les colonnes existantes dans le dataset (`valid_cols`).
3. `tool_params` est forcé à `{}` s'il est absent ou non dictionnaire.
4. `priority` est converti en entier borné dans `[1, 5]` (valeur par défaut `2` en cas d'échec de conversion).
5. L'objet est validé par Pydantic (`Hypothesis.model_validate`).
6. L'hypothèse n'est conservée que si `variables` est non vide et que **toutes** les colonnes citées appartiennent à `valid_cols`.

**Repli automatique (*fallback*)** : si l'appel LLM lève une exception ou si aucune hypothèse valide ne passe le filtrage, HypoLab bascule automatiquement sur le moteur heuristique déterministe.

---

## 3. Cas 2 : Aucun LLM n'est disponible (mode heuristique déterministe)

Lorsqu'aucun LLM n'est configuré, que `use_llm=False` ou que l'appel LLM a échoué, HypoLab appelle `generate_heuristic_hypotheses()` (`hypolab/agents/theorist.py`).

> **Remarque** : ce générateur est 100 % déterministe et s'appuie uniquement sur le `DatasetProfile` ; le paramètre `research_goal` n'est pas utilisé.

### 3.1 Classification préalable des colonnes

À partir de `profile.columns`, les colonnes sont réparties en groupes :
- **`num_cols`** : `semantic_type == "numeric"` et `unique_count > 2`.
- **`bin_cols`** : `semantic_type` dans `("categorical", "boolean", "numeric")` et `unique_count == 2`.
- **`cat_cols`** : `semantic_type == "categorical"` et `2 < unique_count <= 12`.
- **`all_cat_cols`** : concaténation `bin_cols + cat_cols`.

Les colonnes constantes, identifiants, temporelles ou catégorielles à haute cardinalité (`> 12` modalités) sont donc exclues des règles heuristiques.

### 3.2 Les 6 règles de génération (dans l'ordre d'exécution)

Les hypothèses sont ajoutées séquentiellement dans l'ordre ci-dessous jusqu'à atteindre `max_hypotheses` :

#### Règle 1 — Corrélations principales (`test_correlation`, priorité `1`)
- **Source** : les 2 premières paires de `profile.top_correlations` (préalablement triées par $\max(|r|, |\rho|)$ décroissant dans `hypolab/profiler/engine.py`).
- **Direction** : `"positively"` si $r + \rho \ge 0$, sinon `"negatively"`.
- **Forme de relation** : `"monotonic non-linear"` si $|\rho| - |r| > 0.12$, sinon `"linear"`.
- **Confondeur (`control_col`)** : `all_cat_cols[0]` s'il existe, sinon la première colonne de `num_cols` distincte de `col_a` et `col_b` (ou `None`).
- **Paramètres** : `{"col_x": col_a, "col_y": col_b, "control_col": confounder_col}`.

#### Règle 2 — Comparaisons de 2 groupes (`compare_two_groups`, priorité `2`)
- **Déclencheur** : pour chaque colonne binaire `bcol` dans `bin_cols`, associe la **première** colonne `ncol` de `num_cols` telle que `bcol.name != ncol.name` (une seule hypothèse générée par colonne binaire grâce au `break`).
- **Confondeur** : la première autre colonne de `num_cols` différente de `ncol` (si disponible).
- **Paramètres** : `{"group_col": bcol.name, "target_col": ncol.name}`.

#### Règle 3 — Comparaisons multi-groupes (`compare_multiple_groups`, priorité `2`)
- **Déclencheur** : pour chaque colonne catégorielle `ccol` dans `cat_cols` (3 à 12 modalités), associe la **première** colonne `ncol` de `num_cols` (une seule hypothèse par colonne catégorielle).
- **Confondeurs** : `[]`.
- **Paramètres** : `{"group_col": ccol.name, "target_col": ncol.name}` (ANOVA / Kruskal-Wallis).

#### Règle 4 — Association catégorielle (`test_categorical_association`, priorité `3`)
- **Déclencheur** : `len(all_cat_cols) >= 2` et quota `max_hypotheses` non atteint.
- **Variables** : les deux premières colonnes `all_cat_cols[0]` et `all_cat_cols[1]`.
- **Confondeurs** : `[]`.
- **Paramètres** : `{"col_a": ca.name, "col_b": cb.name}` (test du $\chi^2$ / Fisher exact).

#### Règle 5 — Régression multivariée (`multivariate_regression`, priorité `2`)
- **Déclencheur** : `len(num_cols) >= 2` et quota `max_hypotheses` non atteint.
- **Cible (`target`)** : la dernière colonne numérique (`num_cols[-1]`).
- **Prédicteurs (`features`)** : jusqu'aux 3 premières autres colonnes numériques (`num_cols[:-1][:3]`), auxquelles s'ajoute `all_cat_cols[0]` s'il existe.
- **Confondeurs** : `predictor_cols[1:]`.
- **Paramètres** : `{"target": target_col, "features": predictor_cols, "family": "auto"}`.

#### Règle 6 — Détection de courbure quadratique en sandbox (`python_sandbox`, priorité `3`)
- **Déclencheur** : `len(num_cols) >= 2` et quota `max_hypotheses` non atteint.
- **Variables** : les deux premières colonnes numériques (`num_cols[0]` en $x$, `num_cols[1]` en $y$).
- **Code exécuté** : script Python pré-généré ajustant deux modèles OLS avec `statsmodels` :
  - linéaire : $y = \beta_0 + \beta_1 x$ ;
  - quadratique : $y = \beta_0 + \beta_1 x + \beta_2 x^2$.
- **Métriques** : $t$-statistique et $p$-valeur du terme quadratique $\beta_2$, gain de $R^2$ ajusté ($\Delta R^2_{\text{adj}}$) et nuage de points Plotly.

---

## 4. Comparatif synthétique

| Critère | Mode LLM disponible | Mode sans LLM (heuristique) |
| :--- | :--- | :--- |
| **Prise en compte du `research_goal`** | Oui | Non |
| **Utilisation des alertes qualité** | Oui (incluses dans `to_llm_summary()`) | Non |
| **Sélection des variables et méthodes** | Contextuelle et flexible | Séquentielle selon 6 règles fixes |
| **Usage de `python_sandbox`** | Code libre généré par le LLM | Gabarit fixe de régression quadratique uniquement |
| **Déterminisme** | Faible variance (`temperature=0.2`) | 100 % déterministe |
| **Résilience** | Validation Pydantic + repli automatique vers l'heuristique | Autonome, sans réseau ni clé API |
