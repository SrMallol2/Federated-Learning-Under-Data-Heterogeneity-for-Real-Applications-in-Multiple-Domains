**Use Case 1, Health**

(Dates are aproximate)

This journal is done in order to help me with the task of writting the final tfg redaction.

First I would like to explain that the current main idea is to study the accuracy vs communication costs of FL-(KD) enviroments.
This motivation is open to change if during the investigation i find something a bit more niche worth the investigation.

I am enclined towards a medical/network field. Use case 1 will be health related, while use case 2 will be health related. Use Case 1 will probably serve as an introduction to my tfg


27/2/26 

For Use Case 1, I would like to work with a simple dataset, health related, with a classification problem. Ideally no image processing, only raw data (for the moment)

Currently I am using the sklearn, breast cancer dataset as it is simple and ready to use.

For the centralized model i am also currently using the sklearn MLPClassifier, as is the one with the highest precision, and the one that makes the most sense to use in a FL-KD enviroment.

The problem with this model is the low interpretability it has as for health related models, the models must have high interpretability as it is needed to understand the reasoning of the model to make sure the used criteria makes sense in a medical way.

Also I need to switch to Torch, as sklearn library was just for testing

4/3/26

Already moved to torch. Also found a new dataset, small one, 130_diabates. Basically it contains data from about 100k encounters from different patients in 130 diferent US hospitals with diabetes as a diagnostic. The goal is to predict if a patient is going to be readmited or not into the hospital in the next 30 days, as it is dangerous to send someone home who is not well enough and it is expensive to readmit them once they are worse.

I am first going to focus on the data preprocessing and the central model. At the moment i am not going to do much of feature extraction, but it is also interesting to consider. All data is tabular, which also makes the task easier, as for the moment i am not interested in doing any kind of image processing. Also, for the moment as i am working with a small amount of features i do not consider using any kind of feature reduction.

15/3/26

I've mostly finalized the datapreprocessing and the central model. The results of this are under the state of the art paper. This makes sense as i lack the feature extraction. Also i am starting with the federated setup. Currently i am doing a simple FedAvg. 

I am already encountering some problems, if i want to do KD. The main problem is that in order to create synthethic data, it not only must be correct by the joint probabilities of the distribution, but also it must make sense in a medical way, as if it does not it produces a strong bias i can not accept.

I should make a deep look into the distribution of the dataset in order to understand a bit more on how to manuver around it.

Also it is important to account for the geographic biases, as i am not sure how much impact does it have the health difference between the US and Europe for example.

Also i should be reading a lot more and maybe do less work. Some of my issues are probably already solved, which makes this work a bit redundant.

23/3/26

Okey, now i've read a bit more of the papers already using the 130 dataset. I've done some changes in data preparation and added some feature engeneering. Basically I've kept the A1C measurement. Basically because in the paper "Impact of HbA1c Measurement on Hospital Readmission Rates by Strack" it concludes that although stastically it is not significant if it is low or high, the presence of it, indicates a better care for the patient, therefore, less rate of readmission (Only for diabetes).

Also done some feature engineering, from the paper "Prediction of Hospital Readmission using Federated Learning by Sazdov". The same way they do in the paper i create "service utilization" that is the num of times a patient has used the hospital's services. inpatient visits + outpatient visits + emergency visits. Medications count conuts the changes of medications, regardless if it s to a higher or lower dose, since the patien was admitted.

I am not sure what to do about medical speciality. In Strack's paper it is proven that ´specialty × age and specialty × time in hospital´ are stastistically significant. But the main problem is that it almost doubles the amounts of features. I am not worried for overfitting (160 features is still a very managable number) but i am more worried about the FL setup, since any redundant bits i can safe will be welcomed.

I think i will only keep the most significant medical specialities

Talking about FL. I already liked the topic of discussing the accuracy vs comunication costs, and maybe i can prove that by given dirichlet distributions, the fed avg, performs worse that the k gen for any alpha, that is to discuss three things, accuracy, communication and heterogeny of clients. With this i can work on a good thesis.

For the FedAvg, it is mostly finnished. Is just under SOTA, but i do not know exactly why as i have copied or even added some extra features.

The main problem is how i deal with the issues relating to the gen in health. But i believe that if i create vectors I could study the biases they produce under the health constraints. I start implement the fed gen from the paper "Data-Free Knowledge Distillation for Heterogeneous Federated Learnnig by Zhu". Once this is completed i will do a back up and make an analysis regarding the MB of data vs the hetereogenic of data vs the performance of the model for the 3 main cases (Central, FedAvg, FedGen). I believe that my thesis should prove that FedGen is better than FedAvg on the frontier. We will see.

31/3/26

In order to proceed with fedgen i should study more the distribution of the data. This is something I should have done to begin wiht but here we are

Alright. Although there are some things that i do not like, there is some restrictions i can add to the vectors of the generators in order to force them to make them more similar to the real ones.

I think i am laking some measurements of compararissons. Both FedAvg and FedGen are already done, and i should start to compare them. I should add more statisticall measurements.

8/4/26

The regularization of the generator is not working as expected. I've encountered that the latent space does not produce a nucleoid, but rather an inverted parabola. This produces a median on the empty part of it. Meaning that the generated output does not comply with an actual patient.

I have different ways of fixing this or to directly do not do it. I've considered the medioid, i'll test it. Also Zhou's paper already has a regularization, but is purely statisticall in order to avoid the generator to converging. 

Also i am struggling with the measurement of MB. Something does not add up.

Surprised by the results of the partial models.


Okey i am still working on this, this is a big mess. 

To beggin with the noise creation, in zhu's implementation there is a contradiction, it states that the noise comes from Gaussian, but the actual implementation is a Uniform.

I believe that it is best to keep the current Gaussian, as it avoids the bias from Uniform, it is centered at 0.

Secondly I have found some things that do not match between models.

10/4/26

Nice, the problem with the models accuracy comes from the minimum requirement partiton. In order to train the models, since the amount of data is limmited, I added a condtition where there was a need for a mininum amount of dataset. This has taken me an absourd amount of time, and i should have known that that would affect the final distribution. The main problem comes that if i add the condition, the partition is redone until i found one where that condition is match, and that results in a partition of high homogenity, regardless of the alpha.

Now i need to discuss wheter to keep that condition or to do it fully random by alpha (knowing fully well that will affect the model performance)


17/4/26

Okey, rn i am testing lots of different models, both from partial and full models. Partial models have the advantatge of only sharing the final prediction layer, which makes it more secure and with les MBs. The full models on the other hand tend to perform better

21/4/26 

Mostly everything is done for UC1, just need to organize everything and it is almost done

