# Sentiment Analysis
Sentiment analysis deals in analysing the feelings, attitudes, emotions or opinions depending on
different applications. There are many challenges involved in Sentiment Analysis, especially getting
the labelled data for training the model. Here comes the crowdsourcing platforms for the rescue.
However the issues involved with crowdsourced data are :
1. Heterogeneous crowd workers
2. Poor quality and biased labels
3. Spam
4. Adversarial behaviour

### Data Extraction for Gold Standard dataset

To generate the samples, sample function from the pandas library is used and a sample of 1000
records has been extracted from the gold-standard data. After extracting the sample, label
distributions are calculated to verify the equal distribution between the classes. As the distribution
of labels is symmetric, it is satisfying the equal distribution criteria eliminating bias.

### Data extraction and processing the labels on mturk dataset:
To generate the same set observation as gold sample generated in Task 1, ID of the movie is used as
key to extract all the observations from mturk dataset. This resulted in a dataset with movies
matching the movies in gold-standard sample and all the corresponding votes given by different
users (annotators).
The label for each movie has been calculated based on majority voting. To do in python, below steps
are followed:
1. Converted the labels to numeric values, 1 for ‘pos’ label and -1 for ‘neg’ label.
2. The dataset is grouped on ‘id’, calculating mean for rest of the fields. By using mean, the
feature values belonging to the same movie review id remains same, whereas the label
mean results in a number in the range of -1 to 1.
3. Above label values are converted to positive (1) or negative (0) labels using a threshold
value of 0.
4. If there is a tie between number of positive and negative labels, above label calculation
(mean) results in zero. We have considered the positive label for tie.

### Processing the labels using David & Skene method:
In this method, David & Skene algorithm is applied on the mturk dataset to find the polarities of the
movie reviews using crowdsourcing data.

## Performance Evaluation 

#### Part 1 - Model training and testing using gold-standard data
The steps involved :
1. Using the data extracted (as detailed above) built a Decision Tree classification model using
the sklearn library.
2. This model is trained and then tested against the test data provided.

#### Part 2 - Model training and testing using m_turk data
The steps involved :
1. Using the data extracted (as detailed above) built a Decision Tree classification model using
the sklearn library.
2. This model is trained and then tested against the test data provided.

#### Part 3 - Model training and testing using m_turk data using David & Skene method
The steps involved :
1. Using the data extracted (as detailed above) built a David & Skene method for extracting the
labels based on the credibility of workers.
2. built a Decision Tree classification model using the sklearn library.
3. This model is trained on the data generated from the step 1 and then tested against the test
data provided.

#### David & Skene Algorithm :
1. Initialize the estimated polarity for each tweet using the majority vote.
2. Estimate the reliability of each worker based on the estimated polarity in step 1
3. Re-estimate the polarity. (with the new weights to the worker based on their reliability)
4. Go to step 2 until convergence
