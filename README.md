# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PATEL PRIYANSHU BAKULBHAI

*INTERN ID*: CTIS8454

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

This folder provides a lightweight, browser-based demo that mirrors the CIFAR-10 image-classification workflow. The original notebook (`TASK-3.ipynb`) documents a TensorFlow CNN; the web app offers a fast, interactive prototype using a nearest-centroid classifier over compact image features.

Key Features
- Interactive dashboard: prediction panel, per-sample probabilities, gallery, and confusion matrix.
- Evaluation: top-1 accuracy, macro F1, per-class breakdown, top-k accuracy, and average confidence.
- Self-contained demo: the UI computes predictions and metrics in the browser for immediate inspection.

Files
- `index.html`, `styles.css`, `app.js` — the static web demo.
- `TASK-3.ipynb` — reference notebook for the CNN experiment.

Quick Run
Open the `TASK-3` folder in a browser or serve it locally:

```bash
python -m http.server 8000
# then visit http://localhost:8000/TASK-3/
```

Notes
- The browser model is educational and deterministic; replace with TensorFlow.js or a backend service for production-grade inference.
- Use the notebook for training on real CIFAR-10 data and exporting models if needed.

Contact
Open an issue for questions or to suggest enhancements.

#OUTPUT

<img width="1918" height="982" alt="Image" src="https://github.com/user-attachments/assets/c9573bc7-ff01-4461-a093-842e4f06b8ac" />

<img width="1917" height="1007" alt="Image" src="https://github.com/user-attachments/assets/6c8a78ee-65f3-4db8-b2a1-ada7f2c11409" />

<img width="1182" height="902" alt="Image" src="https://github.com/user-attachments/assets/593058b5-c86d-433f-a571-cea1047ca474" />
