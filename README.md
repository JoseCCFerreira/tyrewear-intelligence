# TyreWear Intelligence

Premium portfolio landing page and technical support documentation for a tyre analytics project.

The project explains a local-first analytics platform for tyre comparison, tread-depth statistics,
machine learning, dashboarding and AI-assisted decision support.

## Files

```text
index.html
style.css
technical_support.html
scripts/test_deep_learning.py
scripts/create_snapshot.py
snapshots/
```

## Run The HTML

Open directly in the browser:

```bash
open index.html
open technical_support.html
```

Or serve locally:

```bash
python3 -m http.server 8570
```

Then open:

```text
http://localhost:8570/index.html
http://localhost:8570/technical_support.html
```

## Test PyTorch And TensorFlow

The deep learning smoke test creates synthetic tyre data and trains one tiny PyTorch model and one
tiny TensorFlow model to predict tyre life in kilometres.

```bash
python3 scripts/test_deep_learning.py
```

Output is written to:

```text
snapshots/deep_learning_test_results.json
```

## Create Snapshot

Run the full data, statistics, ML, deep learning and snapshot pipeline:

```bash
python3 scripts/run_full_pipeline.py
```

```bash
python3 scripts/create_snapshot.py
```

## Streamlit Results App

```bash
streamlit run app.py --server.port 8580
```

Then open:

```text
http://localhost:8580
```

The app shows raw data, cleaning, distributions, regional/country time trends, hypothesis tests,
classic ML, clusters, PyTorch/TensorFlow results and generated outputs.

Outputs:

```text
snapshots/project_snapshot.json
snapshots/project_snapshot.md
```

## Git

```bash
git init
git add .
git commit -m "feat: create tyrewear intelligence landing"
```

Add a remote before pushing:

```bash
git remote add origin <repo-url>
git push -u origin main
```
