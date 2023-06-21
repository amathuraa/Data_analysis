# Data_analysis
Personal Projects Relating to Data Analysis
In this Repo I have several projects that I have worked on previously including the source code, raw data, and the output/conclusion from project

# [HKTV MALL ANALYSIS](https://github.com/amathuraa/Data_analysis/tree/main/HKTV_MALL_ANALYSIS)
The objective of this project was to investigate the correlation between weather and clothing sales, specifically winter clothing sales in Hong Kong. The project aimed to determine whether clothing sales depended on external factors such as weather in addition to price. To achieve this objective, the project used data from HKTV mall, the largest 24-hour online shopping mall in Hong Kong, and the Hong Kong Observatory to provide weather trends for the years 2015-2022.

The results of the project showed that there was a steady trend of increased sales of winter clothing when the weather was usually around 20-25 degrees Celsius, during the months of October, November, and December, before the weather gets cooler. On the other hand, there was a steady trend of having the least number of sales during the months of May, June, and July.<br>

An Example graph is <br>

![image](https://github.com/amathuraa/Data_analysis/assets/106806420/9a59ee7f-9b71-437e-8d70-fd83fa5b8f30)

Based on these findings, the project recommended that HKTV mall releases their winter clothing before the cooler months start, around October, and discounts their winter clothing during the months of summer, May and June, to have steady sales throughout the year. With discounted winter attire, people in Hong Kong would have more incentive to buy the clothing, thinking of buying the winter clothes while it's on sale for the upcoming winter instead of buying it at full price.

Overall, this project demonstrates the value of using data analytics to gain insights into customer behavior and make data-driven recommendations that can improve business performance.

# [Car_price_modelling](https://github.com/amathuraa/Data_analysis/tree/main/Car_price_modelling)
The objective of this project was to create a model to predict the price of a car using data from a CSV file containing car data. The data included both numerical and categorical variables, which required preprocessing steps such as one-hot encoding to deal with categorical variables.

To evaluate the performance of the model, k-fold cross-validation was used to split the data into training and testing sets. The greedy algorithm was implemented to select the most important features for the linear model to improve the accuracy of the predictions.

The linear model was trained on the preprocessed data, using the selected features and k-fold cross-validation, to predict the price of the car. The model was evaluated using metrics such as mean squared error and R-squared to assess its performance.

Overall, this project demonstrated the importance of preprocessing data, feature selection, and model evaluation in creating an accurate predictive model. It also highlighted the benefits of using machine learning techniques such as k-fold cross-validation and the greedy algorithm to improve the accuracy of the model.

# [Handwriting Recognition](https://github.com/amathuraa/Data_analysis/tree/main/Handwriting_recognition)
The goal of this project was to compare the effectiveness of decision trees and multinomial logistic regression models in recognizing handwritten digits using the sklearn digits dataset. The dataset consists of 8x8 pixel images of handwritten digits from 0 to 9.
<br>
![image](https://github.com/amathuraa/Data_analysis/assets/106806420/d33cc8ca-a112-4887-8224-a736431f32e8)
<br>

The dataset was preprocessed by scaling the features and splitting it into training and testing sets. Two models were then trained on the data: a decision tree model and a multinomial logistic regression model. The models were evaluated on the testing set using accuracy as the performance metric.

The decision tree model achieved an accuracy of 0.9, while the multinomial logistic regression model achieved an accuracy of 0.97. This suggests that the multinomial logistic regression model is more effective in recognizing handwritten digits than the decision tree model.

This project highlights the importance of comparing different machine learning algorithms to determine the most effective approach for a specific task. It also demonstrates the usefulness of performance metrics such as accuracy in evaluating model performance.

# [PCA Analysis](https://github.com/amathuraa/Data_analysis/tree/main/PCA_analysis)
The objective of this project was to use PCA analysis and Kmeans clustering to recognize handwritten digits using the sklearn digits dataset. The dataset consists of 8x8 pixel images of handwritten digits from 0 to 9.

The dataset was preprocessed by scaling the features and reducing the dimensionality using PCA analysis. Kmeans clustering was then applied to the reduced dataset to cluster the handwritten digits.

The performance of the Kmeans clustering model was evaluated using a confusion matrix to determine the accuracy of the model. The confusion matrix indicated that the accuracy was lower than other similar projects.
<br>
This is the confusion matrix -<br>
![image](https://github.com/amathuraa/Data_analysis/assets/106806420/3a652137-f12f-4e32-be5c-13fcc0fb5f7b)

This project highlights the importance of using dimensionality reduction techniques such as PCA analysis and clustering algorithms such as Kmeans to recognize handwritten digits. It also demonstrates the usefulness of performance metrics such as confusion matrix in evaluating the accuracy of the model.

# [Trading Strategy](https://github.com/amathuraa/Data_analysis/tree/main/Trading)
The objective of this project was to develop a Profit and Loss (PnL) program that uses previous bitcoin values and linear regression to identify an optimal trading strategy and display the potential returns of the best subset of categories to select. The program aimed to maximize profits for a given investment by selecting the most profitable trading strategy.

The program used historical bitcoin values to train a linear regression model to predict future prices. The model was then used to identify the optimal trading strategy by analyzing the potential returns of various subsets of categories. The ideal subset of categories was determined using a greedy algorithm.

The program displayed the potential returns of the best subset of categories to select, allowing the user to make informed trading decisions based on the predicted returns. The program also provided visualizations of the predicted bitcoin prices over time to aid in decision-making.

The final product was 50 iterations of the ideal trading strategy and back testing generated the folling histogram - <br>
![image](https://github.com/amathuraa/Data_analysis/assets/106806420/86cec52f-d4a5-4473-9fd1-bf91eae6ccf3)

Overall, this project demonstrates the value of using machine learning techniques such as linear regression and greedy algorithms to develop profitable trading strategies. It also highlights the importance of displaying potential returns to aid in decision-making and improve trading outcomes.

# [Sentiment Classification](https://github.com/amathuraa/Data_analysis/tree/main/sentiment_classification)
The objective of this project was to build a sentiment calculator for Amazon product reviews using AdaBoost (decision trees) and logistic regression models. The goal was to determine if a review was positive or negative based on the sentiment of individual words used in the review.

The project used a dataset of Amazon product reviews, which were preprocessed to extract the text and sentiment labels. The text was then transformed into features using a bag-of-words approach, which represented the frequency of individual words in the review. The sentiment labels were binary, with 1 indicating a positive review and 0 indicating a negative review.

Two models were trained on the preprocessed data: an AdaBoost model using decision trees and a logistic regression model. The AdaBoost model was used to identify the most informative features, which were then used to train the logistic regression model.

The accuracy of the models was evaluated using a testing dataset. The logistic regression model achieved an accuracy of 0.79 while the AdaBoost model achieved an accuracy of 0.66. However, cross-validation tests proved that both models can reach an accuracy of 0.99. But the processing time increased significantly, and there is a risk of overfitting.

![image](https://github.com/amathuraa/Data_analysis/assets/106806420/6a61baa7-bf31-4320-a5e6-7725808d32e8)<br>
The above proves that despite training values reaching 0.99, as the depth increases the testing accuracy decreases thus proving the model is overfitting<br>
![image](https://github.com/amathuraa/Data_analysis/assets/106806420/22bc6856-1d2e-464e-b7f2-601aa1e9af07)
Similarly for Random Forest models, the testing dataset reaches 0.99 however is overfit as evident from the scores for the testing accuracy.


The output of the sentiment calculator could determine if a review was positive or negative based on the sentiment of individual words in the review. The sentiment calculator could also provide an overall sentiment score for a product based on the sentiment of all the reviews.

Overall, this project highlights the importance of comparing different machine learning algorithms and using cross-validation techniques to determine the most effective approach in building sentiment calculators for product reviews. It also demonstrates the usefulness of performance metrics such as accuracy in evaluating model performance and the potential risks of overfitting.
