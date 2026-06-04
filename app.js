const classNames = [
  'Airplane',
  'Automobile',
  'Bird',
  'Cat',
  'Deer',
  'Dog',
  'Frog',
  'Horse',
  'Ship',
  'Truck',
];

const classSpecs = {
  Airplane: {
    accent: '#8bb7ff',
    accent2: '#3f6fd8',
    accent3: '#d6e6ff',
    bgA: '#16233b',
    bgB: '#0c1426',
    pattern: 'wings',
    center: [0.22, 0.28, 0.84, 0.18, 0.26, 0.80],
  },
  Automobile: {
    accent: '#ffd479',
    accent2: '#ce7a1e',
    accent3: '#fff1c8',
    bgA: '#301f10',
    bgB: '#120c08',
    pattern: 'road',
    center: [0.82, 0.38, 0.16, 0.70, 0.44, 0.58],
  },
  Bird: {
    accent: '#82e8bd',
    accent2: '#3bb77c',
    accent3: '#d7ffe9',
    bgA: '#13261f',
    bgB: '#09160f',
    pattern: 'diagonal',
    center: [0.44, 0.78, 0.26, 0.34, 0.66, 0.76],
  },
  Cat: {
    accent: '#f5a7c4',
    accent2: '#bb5c8b',
    accent3: '#ffe2ee',
    bgA: '#2a1721',
    bgB: '#120b10',
    pattern: 'face',
    center: [0.56, 0.24, 0.48, 0.72, 0.42, 0.64],
  },
  Deer: {
    accent: '#cfb279',
    accent2: '#8f6d3d',
    accent3: '#f2e6cb',
    bgA: '#261b12',
    bgB: '#100b07',
    pattern: 'forest',
    center: [0.36, 0.68, 0.30, 0.58, 0.80, 0.72],
  },
  Dog: {
    accent: '#ffbc8d',
    accent2: '#c96b3e',
    accent3: '#ffe0cc',
    bgA: '#2a1a14',
    bgB: '#120b08',
    pattern: 'face',
    center: [0.60, 0.36, 0.24, 0.76, 0.50, 0.66],
  },
  Frog: {
    accent: '#9ef08a',
    accent2: '#4cb44b',
    accent3: '#ebffd8',
    bgA: '#142716',
    bgB: '#08100a',
    pattern: 'spots',
    center: [0.26, 0.66, 0.30, 0.42, 0.84, 0.82],
  },
  Horse: {
    accent: '#d0b5ff',
    accent2: '#7b4fcc',
    accent3: '#f0e6ff',
    bgA: '#20172c',
    bgB: '#0f0a15',
    pattern: 'mane',
    center: [0.48, 0.30, 0.56, 0.78, 0.56, 0.74],
  },
  Ship: {
    accent: '#83dfff',
    accent2: '#2a86c8',
    accent3: '#d9f5ff',
    bgA: '#102638',
    bgB: '#08121d',
    pattern: 'waves',
    center: [0.20, 0.38, 0.86, 0.24, 0.34, 0.88],
  },
  Truck: {
    accent: '#f58f8f',
    accent2: '#c24949',
    accent3: '#ffd7d7',
    bgA: '#2a1414',
    bgB: '#110909',
    pattern: 'road',
    center: [0.84, 0.28, 0.20, 0.68, 0.48, 0.76],
  },
};

const featureLabels = [
  'color-energy',
  'edge-density',
  'texture-complexity',
  'shape-contrast',
  'symmetry',
  'foreground-focus',
];

const elements = {
  statsGrid: document.getElementById('statsGrid'),
  classFilter: document.getElementById('classFilter'),
  sampleSelect: document.getElementById('sampleSelect'),
  rerunButton: document.getElementById('rerunButton'),
  sampleTitle: document.getElementById('sampleTitle'),
  confidencePill: document.getElementById('confidencePill'),
  imageStage: document.getElementById('imageStage'),
  actualLabel: document.getElementById('actualLabel'),
  predictedLabel: document.getElementById('predictedLabel'),
  featureTags: document.getElementById('featureTags'),
  probabilityBox: document.getElementById('probabilityBox'),
  metricsGrid: document.getElementById('metricsGrid'),
  classBreakdown: document.getElementById('classBreakdown'),
  confusionMatrix: document.getElementById('confusionMatrix'),
  gallery: document.getElementById('gallery'),
};

const dataset = buildDataset();
let model = trainModel(dataset.train);
let evaluation = evaluateModel(model, dataset.test);
let activeFilter = 'All classes';
let activeSampleId = dataset.test[0].id;

populateFilters();
renderApp();

elements.classFilter.addEventListener('change', (event) => {
  activeFilter = event.target.value;
  const visible = getVisibleTestSamples();
  activeSampleId = visible[0]?.id ?? dataset.test[0].id;
  syncSampleSelect();
  renderApp();
});

elements.sampleSelect.addEventListener('change', (event) => {
  activeSampleId = event.target.value;
  renderSelectedSample();
  renderGallery();
});

elements.rerunButton.addEventListener('click', () => {
  model = trainModel(dataset.train);
  evaluation = evaluateModel(model, dataset.test);
  renderApp();
});

function buildDataset() {
  const train = [];
  const test = [];

  classNames.forEach((className, classIndex) => {
    for (let i = 0; i < 5; i += 1) {
      train.push(createSample(className, classIndex, i, 'train', jitterSeed(classIndex, i), 0.035));
    }

    for (let i = 0; i < 3; i += 1) {
      const hardExample = className === 'Cat' && i === 0;
      test.push(createSample(className, classIndex, i, 'test', jitterSeed(classIndex, i + 7), hardExample ? 0.08 : 0.06, hardExample));
    }
  });

  return { train, test };
}

function createSample(className, classIndex, sampleIndex, split, seed, amplitude, hardExample = false) {
  const spec = classSpecs[className];
  const random = mulberry32(seed);
  const features = spec.center.map((value, index) => {
    const noise = (random() - 0.5) * (amplitude - index * 0.004);
    return clamp(value + noise, 0.02, 0.98);
  });

  return {
    id: `${split}-${classIndex + 1}-${sampleIndex + 1}`,
    label: className,
    classIndex,
    sampleIndex,
    split,
    hardExample,
    seed,
    features,
    spec,
  };
}

function trainModel(samples) {
  const centroids = new Map();
  const counts = new Map();

  classNames.forEach((className) => {
    centroids.set(className, Array(featureLabels.length).fill(0));
    counts.set(className, 0);
  });

  samples.forEach((sample) => {
    const centroid = centroids.get(sample.label);
    sample.features.forEach((value, index) => {
      centroid[index] += value;
    });
    counts.set(sample.label, counts.get(sample.label) + 1);
  });

  classNames.forEach((className) => {
    const centroid = centroids.get(className);
    const count = counts.get(className) || 1;
    centroids.set(className, centroid.map((value) => value / count));
  });

  return { centroids };
}

function classify(sample, currentModel) {
  const ranking = classNames.map((className) => {
    const centroid = currentModel.centroids.get(className);
    const distance = euclideanDistance(sample.features, centroid);
    return { className, distance };
  }).sort((left, right) => left.distance - right.distance);

  const logits = ranking.map((entry) => Math.exp(-entry.distance * 3.4));
  const total = logits.reduce((sum, value) => sum + value, 0);
  const probabilities = Object.fromEntries(ranking.map((entry, index) => [entry.className, logits[index] / total]));

  return {
    label: ranking[0].className,
    confidence: probabilities[ranking[0].className],
    probabilities,
    ranking,
    top3: ranking.slice(0, 3).map((entry) => entry.className),
  };
}

function evaluateModel(currentModel, testSamples) {
  const confusion = Object.fromEntries(classNames.map((actual) => [actual, Object.fromEntries(classNames.map((predicted) => [predicted, 0]))]));
  const perClass = Object.fromEntries(classNames.map((className) => [className, { correct: 0, support: 0 }]));
  const predictions = [];

  let correct = 0;

  testSamples.forEach((sample) => {
    const prediction = classify(sample, currentModel);
    predictions.push(prediction);
    confusion[sample.label][prediction.label] += 1;
    perClass[sample.label].support += 1;
    if (prediction.label === sample.label) {
      correct += 1;
      perClass[sample.label].correct += 1;
    }
  });

  const accuracy = correct / testSamples.length;
  const averageConfidence = average(predictions.map((prediction) => prediction.confidence));
  const top3Accuracy = average(predictions.map((prediction, index) => prediction.top3.includes(testSamples[index].label) ? 1 : 0));

  let precisionSum = 0;
  let recallSum = 0;
  let f1Sum = 0;

  classNames.forEach((className) => {
    const tp = confusion[className][className];
    const fp = classNames.reduce((sum, actualClass) => sum + (actualClass === className ? 0 : confusion[actualClass][className]), 0);
    const fn = classNames.reduce((sum, predictedClass) => sum + (predictedClass === className ? 0 : confusion[className][predictedClass]), 0);
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    precisionSum += precision;
    recallSum += recall;
    f1Sum += f1;
  });

  return {
    accuracy,
    precision: precisionSum / classNames.length,
    recall: recallSum / classNames.length,
    macroF1: f1Sum / classNames.length,
    top3Accuracy,
    averageConfidence,
    correct,
    confusion,
    perClass,
    predictions,
  };
}

function renderApp() {
  renderStats();
  syncSampleSelect();
  renderSelectedSample();
  renderProbabilityBox();
  renderMetrics();
  renderClassBreakdown();
  renderConfusionMatrix();
  renderGallery();
}

function renderStats() {
  const cards = [
    { label: 'Training samples', value: dataset.train.length, note: 'feature prototypes used to fit the classifier' },
    { label: 'Test samples', value: dataset.test.length, note: 'held-out images used for evaluation' },
    { label: 'Accuracy', value: formatPercent(evaluation.accuracy), note: 'top-1 performance on the test split' },
    { label: 'Macro F1', value: evaluation.macroF1.toFixed(3), note: 'balanced view across classes' },
  ];

  elements.statsGrid.innerHTML = cards.map((card) => `
    <article class="card stat-card">
      <span class="stat-value">${card.value}</span>
      <span class="stat-label">${card.label}</span>
      <p class="metric-subtitle">${card.note}</p>
    </article>
  `).join('');
}

function renderSelectedSample() {
  const sample = getSelectedSample();
  const prediction = classify(sample, model);

  elements.sampleTitle.textContent = `${sample.label} sample ${sample.sampleIndex + 1}`;
  elements.confidencePill.textContent = `Confidence ${(prediction.confidence * 100).toFixed(1)}%`;
  elements.actualLabel.textContent = sample.label;
  elements.predictedLabel.textContent = prediction.label;

  elements.featureTags.innerHTML = sample.features.map((value, index) => `<span class="tag">${featureLabels[index]}: ${value.toFixed(2)}</span>`).join('');
  elements.imageStage.innerHTML = renderImage(sample);
}

function renderProbabilityBox() {
  const sample = getSelectedSample();
  const prediction = classify(sample, model);
  const ranked = classNames
    .map((className) => ({ className, probability: prediction.probabilities[className] }))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, 5);

  elements.probabilityBox.innerHTML = ranked.map((item) => `
    <article class="metric-card">
      <span class="metric-value ${item.className === sample.label ? 'good' : item.className === prediction.label ? 'warn' : 'bad'}">${(item.probability * 100).toFixed(1)}%</span>
      <div class="metric-subtitle">${item.className}</div>
      <p class="metric-note">Probability assigned by the nearest-centroid model</p>
      <div class="bar-track"><div class="bar-fill" style="width:${item.probability * 100}%"></div></div>
    </article>
  `).join('');
}

function renderMetrics() {
  const cards = [
    { label: 'Top-1 accuracy', value: formatPercent(evaluation.accuracy), tone: 'good', note: 'correct label on the first guess' },
    { label: 'Precision', value: evaluation.precision.toFixed(3), tone: 'warn', note: 'how often predicted classes are correct' },
    { label: 'Recall', value: evaluation.recall.toFixed(3), tone: 'warn', note: 'how many actual examples were recovered' },
    { label: 'F1 score', value: evaluation.macroF1.toFixed(3), tone: 'good', note: 'harmonic mean of precision and recall' },
    { label: 'Top-3 accuracy', value: formatPercent(evaluation.top3Accuracy), tone: 'good', note: 'whether the true label appears in the shortlist' },
    { label: 'Avg confidence', value: formatPercent(evaluation.averageConfidence), tone: 'warn', note: 'average certainty from the model' },
  ];

  elements.metricsGrid.innerHTML = cards.map((card) => `
    <article class="metric-card">
      <span class="metric-value ${card.tone}">${card.value}</span>
      <div class="metric-subtitle">${card.label}</div>
      <p class="metric-note">${card.note}</p>
    </article>
  `).join('');
}

function renderClassBreakdown() {
  elements.classBreakdown.innerHTML = classNames.map((className) => {
    const stats = evaluation.perClass[className];
    const accuracy = stats.support === 0 ? 0 : stats.correct / stats.support;

    return `
      <article class="class-card">
        <h3>${className}</h3>
        <span class="class-value ${accuracy >= 0.75 ? 'good' : accuracy >= 0.5 ? 'warn' : 'bad'}">${formatPercent(accuracy)}</span>
        <p class="class-subtitle">${stats.correct} correct of ${stats.support} samples</p>
        <div class="bar-track"><div class="bar-fill" style="width:${accuracy * 100}%"></div></div>
      </article>
    `;
  }).join('');
}

function renderConfusionMatrix() {
  const matrix = evaluation.confusion;
  const maxValue = Math.max(...Object.values(matrix).flatMap((row) => Object.values(row)));

  const header = `<tr><th>Actual / Pred</th>${classNames.map((name) => `<th>${shortLabel(name)}</th>`).join('')}</tr>`;
  const rows = classNames.map((actualClass) => {
    const row = matrix[actualClass];
    const cells = classNames.map((predictedClass) => {
      const value = row[predictedClass];
      const opacity = maxValue === 0 ? 0 : 0.10 + (value / maxValue) * 0.75;
      const background = actualClass === predictedClass
        ? `rgba(124, 240, 200, ${opacity})`
        : `rgba(126, 168, 255, ${opacity * 0.78})`;
      return `<td style="background:${background}">${value}</td>`;
    }).join('');

    return `
      <tr>
        <th>${shortLabel(actualClass)}</th>
        ${cells}
      </tr>
    `;
  }).join('');

  elements.confusionMatrix.innerHTML = `
    <table class="matrix-table">
      <thead>${header}</thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderGallery() {
  const visibleSamples = getVisibleTestSamples();

  if (!visibleSamples.length) {
    elements.gallery.innerHTML = '<div class="empty-state">No samples match the current filter.</div>';
    return;
  }

  elements.gallery.innerHTML = visibleSamples.map((sample) => {
    const prediction = classify(sample, model);
    const isCorrect = prediction.label === sample.label;
    return `
      <article class="sample-card ${sample.id === activeSampleId ? 'active' : ''}" data-sample-id="${sample.id}">
        <div class="sample-thumb">
          <div class="thumb-grid">${renderThumb(sample)}</div>
        </div>
        <div>
          <h3>${sample.label}</h3>
          <p class="class-subtitle">Predicted as ${prediction.label}</p>
          <p class="class-subtitle">Confidence ${(prediction.confidence * 100).toFixed(1)}%</p>
          <div class="sample-chip ${isCorrect ? 'correct' : 'wrong'}">${isCorrect ? 'Correct' : 'Misclassified'}</div>
        </div>
      </article>
    `;
  }).join('');

  elements.gallery.querySelectorAll('[data-sample-id]').forEach((card) => {
    card.addEventListener('click', () => {
      activeSampleId = card.dataset.sampleId;
      syncSampleSelect();
      renderApp();
    });
  });
}

function populateFilters() {
  elements.classFilter.innerHTML = ['All classes', ...classNames].map((name) => `<option value="${name}">${name}</option>`).join('');
  elements.classFilter.value = activeFilter;
}

function syncSampleSelect() {
  const visibleSamples = getVisibleTestSamples();
  elements.sampleSelect.innerHTML = visibleSamples.map((sample) => `<option value="${sample.id}">${sample.label} · sample ${sample.sampleIndex + 1}</option>`).join('');
  if (!visibleSamples.some((sample) => sample.id === activeSampleId)) {
    activeSampleId = visibleSamples[0]?.id ?? dataset.test[0].id;
  }
  elements.sampleSelect.value = activeSampleId;
}

function getVisibleTestSamples() {
  return activeFilter === 'All classes'
    ? dataset.test
    : dataset.test.filter((sample) => sample.label === activeFilter);
}

function getSelectedSample() {
  return dataset.test.find((sample) => sample.id === activeSampleId) ?? dataset.test[0];
}

function renderImage(sample) {
  const pattern = sample.spec.pattern;
  const tiles = [];

  for (let index = 0; index < 12; index += 1) {
    const weight = patternWeight(pattern, index + sample.sampleIndex);
    const toneA = blend(sample.spec.accent, sample.spec.bgA, 0.28 + weight * 0.34);
    const toneB = blend(sample.spec.accent2, sample.spec.bgB, 0.20 + weight * 0.42);
    const sizeClass = index % 5 === 0 ? 'wide' : index % 7 === 0 ? 'tall' : '';
    tiles.push(`<span class="sample-tile ${sizeClass}" style="--tile-a:${toneA}; --tile-b:${toneB};"></span>`);
  }

  return `
    <div class="sample-art" style="--sample-bg-a:${sample.spec.bgA}; --sample-bg-b:${sample.spec.bgB};">
      <div class="sample-badge">${sample.label}</div>
      <div class="sample-grid">${tiles.join('')}</div>
    </div>
  `;
}

function renderThumb(sample) {
  const tiles = [];
  for (let index = 0; index < 16; index += 1) {
    const weight = patternWeight(sample.spec.pattern, index + sample.sampleIndex);
    const toneA = blend(sample.spec.accent, sample.spec.bgA, 0.35 + weight * 0.28);
    const toneB = blend(sample.spec.accent3, sample.spec.bgB, 0.18 + weight * 0.42);
    tiles.push(`<span class="thumb-cell" style="--tile-a:${toneA}; --tile-b:${toneB};"></span>`);
  }
  return tiles.join('');
}

function buildReasons(sample, prediction) {
  const ranked = prediction.ranking.slice(0, 3).map((entry) => `${entry.className} ${entry.distance.toFixed(2)}`);
  return [
    `Closest centroid: ${prediction.label} with ${(prediction.confidence * 100).toFixed(1)}% confidence.`,
    `Top feature cues: ${featureLabels.join(', ')}.`,
    `Nearest competitors: ${ranked.join(' | ')}.`,
  ];
}

function shortLabel(name) {
  return name.length > 6 ? `${name.slice(0, 6)}.` : name;
}

function patternWeight(pattern, index) {
  const base = (index % 4) / 3;
  const waves = Math.sin(index * 0.9) * 0.5 + 0.5;

  switch (pattern) {
    case 'wings': return 0.25 + (index % 2 === 0 ? 0.55 : 0.18);
    case 'road': return 0.18 + base * 0.58;
    case 'diagonal': return 0.2 + waves * 0.5;
    case 'face': return 0.22 + ((index === 5 || index === 6 || index === 9 || index === 10) ? 0.48 : 0.18);
    case 'forest': return 0.24 + ((index % 3) / 2.8);
    case 'spots': return 0.16 + (index % 5 === 0 ? 0.64 : 0.2);
    case 'mane': return 0.2 + (index % 4 === 1 ? 0.58 : 0.16);
    case 'waves': return 0.18 + (Math.sin(index * 0.7) * 0.5 + 0.5) * 0.56;
    default: return 0.24 + base * 0.34;
  }
}

function blend(colorA, colorB, amount) {
  const a = hexToRgb(colorA);
  const b = hexToRgb(colorB);
  const mixed = {
    r: Math.round(a.r * amount + b.r * (1 - amount)),
    g: Math.round(a.g * amount + b.g * (1 - amount)),
    b: Math.round(a.b * amount + b.b * (1 - amount)),
  };
  return `rgb(${mixed.r}, ${mixed.g}, ${mixed.b})`;
}

function hexToRgb(hex) {
  const normalized = hex.replace('#', '');
  const value = normalized.length === 3
    ? normalized.split('').map((char) => char + char).join('')
    : normalized;

  return {
    r: parseInt(value.slice(0, 2), 16),
    g: parseInt(value.slice(2, 4), 16),
    b: parseInt(value.slice(4, 6), 16),
  };
}

function jitterSeed(classIndex, sampleIndex) {
  return classIndex * 97 + sampleIndex * 17 + 11;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function random() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function euclideanDistance(left, right) {
  return Math.sqrt(left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}