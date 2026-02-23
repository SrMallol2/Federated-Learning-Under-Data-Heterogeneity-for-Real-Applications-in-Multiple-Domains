# Notes

Summary of topics for the TFG

## Federated learning

(Content coming soon...)

## Knowledge distribution

You train a bunch of **teachers** on real data, on the **client end**.
These teachers are locally trained models that give good predictions.

The problem is that you cannot share the client's data with the server. So you create a **data generator** that clients can run to:

1. The generator produces synthetic inputs.
2. Each client passes these synthetic inputs through its locally trained model.
3. Clients update their models by minimizing a loss that encourages correct behavior on both real data and synthetic data.
4. Clients send only model updates (not data) to the server.
5. The server aggregates these updates to obtain a global **student model**.

This means the student only uses the output of the teacher to train.

Also, the creation of syntetic data avoids the heterogenic problem of FL, where the different clients behave differently, so they are forced to learn from a "general view" in order to generalize better

### Knowledge Distillation Process:

```
Fake input → [Teacher Model] → soft behaviour
              ↓
Fake input → [Student Model] → output
              ↓
Student learns to match Teacher behaviour
```
