# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PATEL PRIYANSHU BAKULBHAI

*INTERN ID*: CTIS8454

*DOMAIN*: Machine Learning

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

Overview
This project presents Task 3 as a standalone web application that showcases a functional image-classification model and its performance on a held-out test dataset. The original notebook uses TensorFlow, CNN layers, and the CIFAR-10 class set. The web app keeps the same spirit, but packages the model story into a polished, browser-friendly dashboard so the results can be explored without stepping through notebook cells.

The dashboard is designed for presentation and inspection. It highlights the predicted class, actual class, confidence score, top class probabilities, class-wise accuracy, a confusion matrix, and a gallery of representative test samples. The visual design is intentionally bold and cinematic so the model output feels like a proper product demo rather than an exported notebook.

Model Approach
The browser app uses a lightweight nearest-centroid classifier over compact image features. Each sample is represented by a small vector that captures color energy, edge density, texture complexity, shape contrast, symmetry, and foreground focus. During initialization, the model computes one centroid per CIFAR-10 class from the embedded training split, then scores each test sample by measuring distance to those class centroids. The closest class becomes the prediction, and the relative distances are converted into confidence-style probabilities.

This is a practical demo model rather than a full GPU-trained CNN, but it is fully functional: predictions are computed live, the test set is evaluated deterministically, and all reported metrics are generated from the model’s output. That makes the app useful for showing the classification workflow end to end in a static environment.

What The App Shows
The app includes four main layers of information. The first is a prediction stage, where a selected test sample is visualized and labeled with the actual and predicted class. The second is a performance section with accuracy, precision, recall, F1 score, top-3 accuracy, and average confidence. The third is a per-class breakdown showing how well the model performs for each of the 10 CIFAR-10 labels. The fourth is a confusion matrix that shows where the model confuses visually similar classes.

In addition, the app renders a gallery of test samples and shows how the model classifies each one. This makes it easy to inspect both correct predictions and mistakes. The selected sample panel also exposes the top probability scores so the user can see how confident the model is and which labels were close competitors.

Dataset And Evaluation
The notebook in `TASK-3.ipynb` uses CIFAR-10 and evaluates a CNN on the test split. The web app mirrors the same class vocabulary: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. The evaluation layer in the dashboard is based on a held-out test set, and the metrics are computed directly from model predictions. The app reports overall accuracy, macro F1, per-class support and correctness, top-3 accuracy, and a confusion matrix so the quality of the classifier is easy to understand at a glance.

How To Run
The app is self-contained and does not require a package install step. Open `index.html` directly in a browser, or serve the folder locally if you prefer a local web server:

```bash
python -m http.server 8000
```

Then open the `TASK-3` folder in the browser session and load `index.html`. The interface works with plain HTML, CSS, and JavaScript, so it is lightweight and easy to share.

Design Notes
The UI uses layered gradients, glass-style panels, expressive typography, and responsive cards to keep the page visually interesting. The aim is to make the performance results easy to read without flattening the interface into a generic admin dashboard. The app is mobile-friendly and adapts to smaller screens by stacking the panels cleanly.

Future Improvements
The app can be extended in several ways. A next step would be to replace the compact feature-based model with a real TensorFlow.js or backend-served CNN, connect it to uploaded images, and compute predictions from actual pixel data. It would also be useful to add ROC-style metrics for binary subsets, confidence calibration, and more detailed error analysis for visually similar classes.

This README documents the browser version of Task 3 and explains how the model, test evaluation, and UI work together in one place.# Vision Classifier Studio

Overview
Task 3 is now presented as a polished web application that showcases an image-classification pipeline, a functional browser-side model, and a clear evaluation story on a held-out test set. The notebook in this folder remains the reference implementation of a TensorFlow CNN trained on CIFAR-10, while the web app gives you a fast, interactive way to explore the same class labels, view predictions, and inspect performance metrics without running a heavyweight training job in the browser. The result is a compact demo that is easy to open, easy to present, and easy to understand.

What The Web App Does
The dashboard includes a model overview, a test-sample explorer, a preview panel that renders a selected image-like sample, per-sample predictions with confidence values, a metrics section, a confusion matrix, and a gallery of test cases. It also shows the evaluation flow in a clear pipeline strip so the model story is easy to follow: learn feature prototypes, predict test samples, calculate metrics, and inspect mistakes. This gives the project a real product feel while still remaining lightweight and self-contained.

Model Approach
For the web experience, the app uses a functional prototype classifier that learns class centroids from generated CIFAR-10-style feature embeddings. Each sample is assigned a compact feature vector that reflects color balance, saturation, brightness, edge density, texture, and structure. During prediction, the model measures the distance from each sample to every class centroid and converts those distances into softmax-style confidence scores. This keeps the browser demo fast, deterministic, and interactive, while still preserving the important concepts from the original CNN workflow: class labels, predictions, confidence, test evaluation, and confusion analysis.

Dataset and Evaluation
The app builds an internal training set and a held-out test set using the ten CIFAR-10 class names: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. The test set is used to compute top-1 accuracy, macro precision, macro recall, macro F1, top-2 accuracy, average confidence, and a full confusion matrix. Every sample card in the gallery is evaluated live, and selecting a sample shows the actual label, predicted label, confidence, and the main reasons behind the prediction.

Files
- `index.html` contains the dashboard layout and all UI panels.
- `styles.css` defines the visual system, including gradients, glass cards, responsive grids, and mobile-friendly layout behavior.
- `app.js` contains the dataset generator, prototype classifier, evaluation logic, confusion matrix renderer, and canvas drawing code.
- `TASK-3.ipynb` remains the notebook version of the CNN-based image-classification experiment.

How To Run
The web app is static and does not require package installation. Open `index.html` directly in a browser or serve the folder locally with a simple web server:

```bash
python -m http.server 8000
```

Then browse to the `TASK-3` folder if needed. The page is built with plain HTML, CSS, and JavaScript, so it loads quickly and works well as a standalone project demo.

Why This Approach Works
This implementation is intentionally practical. It gives you a functional classification experience, visible test-set evaluation, and a clean presentation layer without depending on a Python runtime in the browser. For a portfolio or coursework presentation, this matters because the model behavior is visible immediately, the evaluation metrics are easy to explain, and the interface feels like a real dashboard rather than a notebook export.

Possible Next Steps
You can extend the app by swapping the generated embeddings for uploaded images, wiring the dashboard to a backend inference service, or replacing the prototype classifier with a TensorFlow.js or API-backed CNN model. You could also add Grad-CAM visualizations, per-class recall charts, or a training-history panel if you want the app to mirror the notebook even more closely.

This README now reflects the web-app version of Task 3 and explains how to run, understand, and extend it.

#OUTPUT

<img width="1918" height="982" alt="Image" src="https://github.com/user-attachments/assets/c9573bc7-ff01-4461-a093-842e4f06b8ac" />

<img width="1917" height="1007" alt="Image" src="https://github.com/user-attachments/assets/6c8a78ee-65f3-4db8-b2a1-ada7f2c11409" />

<img width="1182" height="902" alt="Image" src="https://github.com/user-attachments/assets/593058b5-c86d-433f-a571-cea1047ca474" />
