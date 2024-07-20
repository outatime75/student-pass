

## **2nd Place Solution for the Tuwaiq : Unlocking Potential for Elite Training Programs**


Link to notebook - https://www.kaggle.com/code/karlcini/2nd-place-solution-tuwaiq-challenge 

The objective of this task was to build a predictive model that identifies top candidates likely to successfully complete a training program.

***Context***

-	Overview of the competition - https://www.kaggle.com/competitions/measuring-student-persistence-and-completion-rate/overview  
-	Data - https://www.kaggle.com/competitions/measuring-student-persistence-and-completion-rate/data 

***Overview of the approach***

The bulk of the work consisted of cleaning the data. Given a number of columns had missing data, it was important to select imputation methods that make sense given the data in question. 

Feature engineering was mainly done on program dates and columns with high frequency of missing values were dropped.  New features included a column noting how many times a student would have already applied for the program main category code as well as the combination of program codes and a normalised university score.

A RepeatedStratifiedKFold system was adopted to allow for different stratified selections to be tested with different random states. 

After training and optimizing (using optuna) 4 different classifiers:
-	CatBoost Classifier
-	Gradient Boosting Classifier
-	LightGBM Classifier
-	XGB Classisifer
the predictions were passed through a VotingClassifier using ‘soft’ voting selection mechanism. This enabled an ensemble of classifiers to produce a more robust prediction, by reducing overfitting and where the probabilities predicted by each classifier are averaged, and the class with the highest average probability is chosen.

Finally I also experimented with various thresholds given that the 0.5 default threshold was not ideal. 
 

***Details of the submission***

*Imputation*

Median and most frequent imputation approaches were implemented based on specific characteristics of the data being imputed. The training and testing datasets were merged to carry out this imputation. Grouping was also employed to extract medians for different groups within the column being imputed. 

While initially I tried to impute ‘Education Speciality’, given that this was indicating to be a promising feature, I eventually decided to drop the feature, given the large number of inconsistent entries and reduced CV scores, leading me to suspect that the imputations for this category were not being beneficial. It is suggested that for future reference, this category is standardised by the Academy, to be able to make use of this otherwise important feature. 

*Registration file*

The data in this file was joined to the main dataset and a number of features were extracted, mainly how many times the student would have attempted a course with the same program main code, and a count of the different types of courses applied for. 

*Uncertainty and difficult records*

The submissions and related CV vs LB scores were giving mixed indications. This was possibly due to the relatively small datasets and some difficult records. An analysis of the probability predictions made by the classifiers showed that while some records were being correctly predicted with a high probability a number of others were being predicted with probabilities between 0.45 to 0.55. Most false predictions were being made when a student would have completed a degree but yet fails the program or vice versa, indicating that the ‘Completed Degree’ feature was also an important feature in the dataset. The variation in the probability threshold was hence implemented to account for this uncertainty. 

*Stratification*

It was important to stratify folds with different random seeds to allow for generalization. To minimize variance between fold results a RepeatedStratifiedKFold approach was adopted. Five folds repeated with two different random seeds were used, with a total of 10 different folds for training. 

*Voting Classifier*

Sklearn’s Voting Classifier was used to ensemble the predictions of the chosen classifiers:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html 

A voting classifier works by combining the predictions of multiple base classifiers to make a final prediction. There are different strategies for combining these predictions, including:

*Majority Voting*: In this approach, each base classifier gets one vote, and the class with the most votes is chosen as the final prediction. This is suitable for classifiers that produce discrete outputs, such as class labels.

*Weighted Voting*: Instead of giving each classifier an equal vote, you can assign different weights to each classifier based on their performance or confidence. This allows more influential classifiers to have a greater impact on the final decision.

*Soft Voting*: In soft voting, the probabilities predicted by each classifier are averaged, and the class with the highest average probability is chosen. This approach is suitable when the classifiers provide probability estimates rather than discrete class labels.

*Optimisation using Optuna*

Hyperparameters for the chosen classifiers were individually optimized using Optuna. This allowed each classifier to produce the best probabilities given the allowed combination of hyperparameters that are testing with Optuna. 

