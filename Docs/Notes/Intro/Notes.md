# Notes

Summary of topics for the TFG

## Federated learning

(Content coming soon...)

## Knowledge distribution

### My own wording:

You train a bunch of **teachers** on real data, on the **client end**.
These teachers are heavy models that give good predictions.

The problem is that you cannot share the client's data with the server. So you create a **data generator** that clients can run to:
1. Generate fake data
2. Run it through the teacher model
3. Get soft outputs
4. Send these to train the **student** model on the server

This means the student only uses the output of the teacher to train.

This is done in rounds to try to optimize the 

### Knowledge Distillation Process:

```
Fake input → [Teacher Model] → soft output
              ↓
Fake input → [Student Model] → output
              ↓
Student learns to match Teacher
```
