**Use Case 1, Health**

27/2/26

For Use Case 1, I would like to work with a simple dataset, health related, with a classification problem. Ideally no image processing, only raw data (for the moment)

Currently I am using the sklearn, breast cancer dataset as it is simple and ready to use.

For the centralized model i am also currently using the sklearn MLPClassifier, as is the one with the highest precision, and the one that makes the most sense to use in a FL-KD enviroment.

The problem with this model is the low interpretability it has as for health related models, the models must have high interpretability as it is needed to understand the reasoning of the model to make sure the used criteria makes sense in a medical way.

Also I need to switch to Torch, as sklearn library was just for testing

4/3/26

