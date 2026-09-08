## Predicting Late Shipments

### Overview

10,999 e-commerce orders, and one question: **can you tell in advance which ones will arrive
late?**

Three models are put on the same 2,200-order test set — a majority-class baseline, Gaussian Naive
Bayes, and a tuned neural network. The interesting part is not which one wins. It is that **each
metric picks a different winner**, including two that go to the model which has learned nothing,
and that deciding between them turns out not to be a modelling question at all.

### Technical Implementation
* ****Language****: Python
* ****Libraries****: pandas, scikit-learn (GaussianNB, preprocessing, metrics), TensorFlow / Keras, matplotlib
* ****Tools****: Jupyter Notebook

### The data

E-Commerce Shipping Data — 10,999 orders, 11 features after dropping the row ID. Target is
`Reached.on.Time_Y.N`, where **1 means the order did not arrive on time**. The classes are
imbalanced: 1,313 late against 887 on time in the test set, roughly 60:40.

Features cover the warehouse block, shipping mode, product cost and importance, weight, discount
offered, customer care calls, customer rating, and prior purchases. There is no route, carrier,
distance or date.

### Results

All three evaluated on the same test rows (`test_size=0.2`, `stratify=y`, `random_state=42`).
Precision and recall are for the late class:

| Model | Accuracy | Recall | Precision | F1 | Orders flagged late |
|---|---|---|---|---|---|
| Majority baseline | 59.68% | **1.000** | 0.597 | **0.748** | 2,200 |
| GaussianNB | 64.41% | 0.424 | **0.955** | 0.587 | 582 |
| Neural network (2 × 16, L1/L2 0.001) | **67.2%** | 0.510 | 0.895 | 0.650 | 749 |

![Model comparison](images/model_comparison.png)

### So what

**Every metric picks a different winner.** Accuracy says the network. Recall says the majority
baseline — it catches every late order, because it calls everything late. Precision says
GaussianNB, which is right 95% of the times it speaks up. Even F1, the metric usually reached for
to settle a precision/recall argument, ranks the baseline first at 0.748. A model that has learned
nothing tops two of the four columns. That is the clearest possible sign that no single number
decides this.

**The answer comes from the cost structure, not from the metric.** The real question is what it
costs to be wrong in each direction. If acting on a warning is free, flag every order and be done —
the baseline is optimal and no model is needed. If only a limited number of orders can be chased,
then a model that names about 750 of them and is right roughly 90% of the time is worth far more
than one that names all 2,200 and is right 60% of the time. **"Which model is best" cannot be
answered before knowing the ratio between the cost of missing a late order and the cost of chasing
an on-time one.** That ratio is a business input. It is not in this dataset, and no amount of
modelling will produce it.

**Separately, the ceiling is in the features.** The baseline gets 59.7% for free, Naive Bayes adds
about 5 points, and a neural network with no independence assumption, tuned over 8 configurations,
adds about 3 more and stops. Two model families with very different assumptions landing that close
together says the limit is the information in the columns, not the algorithm. There is no route,
carrier, distance or season here, and `Discount_offered` was the one feature that visibly separated
the classes. More features would move this further than more tuning.

### Why the comparison is valid

The two model notebooks were written independently — they began as separate assignments for a data
mining course, which is why they happen to use an identical split. Before comparing anything I
checked that they were actually measuring the same thing:

- same file, same dropped columns, same target
- same `train_test_split(test_size=0.2, stratify=y, random_state=42)` — identical test rows
- preprocessing fit on the training set only, in both, so neither model saw the test set early

The one deliberate difference is scaling. The network gets `StandardScaler` on the numeric columns
because gradient descent struggles when `Weight_in_gms` runs into the thousands and
`Customer_rating` is 1–5. GaussianNB does not need it — it fits a separate distribution per
feature.

`03_comparison.ipynb` rebuilds all three from the raw file rather than importing results, so the
table above is reproduced end to end.

### Notebooks

| File | What it does |
|---|---|
| `01_naive_bayes.ipynb` | Stratified split, one-hot encoding fit on train only, GaussianNB, confusion matrix, baseline check |
| `02_neural_network.ipynb` | Adds scaling, builds a Keras MLP, searches 8 hyperparameter combinations on a validation set held out of the training data, trains the winner |
| `03_comparison.ipynb` | Rebuilds all three on the same split, reports the full classification report for each, and works out why the metrics disagree |

### Repository Structure

```
shipping-delay-prediction/
├── 01_naive_bayes.ipynb      # GaussianNB + majority-class baseline
├── 02_neural_network.ipynb   # Keras MLP + hyperparameter search
├── 03_comparison.ipynb       # All three on one split, and what it means
├── Train.csv                 # E-Commerce Shipping Data, 10,999 orders
├── images/                   # Chart (generated by 03_comparison.ipynb)
└── README.md
```

### How to Run

```bash
# 1. Clone the repository
git clone https://github.com/JinWanKim98/shipping-delay-prediction.git
cd shipping-delay-prediction

# 2. Install dependencies
pip install pandas scikit-learn tensorflow matplotlib jupyter

# 3. Run the notebooks in order
jupyter notebook
```

Running `03_comparison.ipynb` regenerates the chart in `images/`.

### Limitations

- One train/test split with a fixed seed. Cross-validation would give an error range instead of
  single numbers, and would say whether the gap between the two models is real.
- The decision threshold is fixed at 0.5 throughout. Since the conclusion above turns on the
  precision/recall trade, the honest next step is a threshold sweep once the cost ratio is known —
  the threshold, not the model, is the dial that moves that trade.
- The network's accuracy moves by about a tenth of a point between runs; the baseline and
  GaussianNB figures are exact every time. That is not a machine difference. GaussianNB has a
  closed form — compute a mean and variance per column and stop — so once the split is fixed
  nothing varies. The network is fitted by iterative optimisation, and while `set_seed` pins the
  weight initialisation and the batch shuffling, it does not pin the order in which multi-threaded
  operations accumulate their sums. Floating point addition is not associative, so a different
  summation order gives a slightly different answer. Pinning that as well needs
  `tf.config.experimental.enable_op_determinism()`, at a cost in speed.
- The dataset is a single snapshot with no dates, so nothing here says whether the pattern holds
  over time.
