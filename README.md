# ML_resources
### Here are the resources helpful in learning machine learning.

## Week 1 Resources

Python:-
https://www.youtube.com/playlist?list=PL-osiE80TeTskrapNbzXhwoFUiLCjGgY7 (Videos 2-10)

Numpy:-
https://numpy.org/devdocs/user/quickstart.html (do not get overwhelmed by seeing so much commands :) )

Pandas:-
https://www.w3schools.com/python/pandas/default.asp (till cleaning data)

Matplotlib:-
https://towardsdatascience.com/matplotlib-tutorial-learn-basics-of-pythons-powerful-plotting-library-b5d1b8f67596

Complete Tutorial (Highly Recommended for the beginners):
https://cs231n.github.io/python-numpy-tutorial/

For those who want to work with Jupyter notebooks in VS Code:-
https://youtu.be/h1sAzPojKMg?si=2mrgArY8l3psIME9

https://www.geeksforgeeks.org/python-lists-vs-numpy-arrays/
This is the article that I was talking about.

Here are some additional resources:

For Git:
A beginner-friendly and easy-to-follow video : https://youtu.be/tRZGeaHPoaw?si=06GZKYd83iAvLx8A
Cheat sheet for future reference : https://education.github.com/git-cheat-sheet-education.pdf

For Markdown:
A short 8-min video covering almost everything you will need : https://youtu.be/2JE66WFpaII?si=5eDA-wD6sj0Xv86M
Cheat sheet for quick future reference : https://www.markdownguide.org/cheat-sheet/

P.S. Git is a useful tool used almost everywhere so learn and understand its intricacies properly. Also no need to spend more time than needed in the above resources on markdown.

## Week 2 Resources

https://towardsdatascience.com/what-is-machine-learning-how-i-explain-the-concept-to-a-newcomer-d96f35a5c4f3

https://www.youtube.com/watch?v=xtOg44r6dsE

Now, we will move on with linear regression:

->Start by watching these videos:
Andrew Ng Course
https://youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&si=b9WaafbzNVJTP2EK
Watch from #9 to #20 (Don't worry : these are of average length of 10min only and provide good intuition)

Read the summary here:
https://www.geeksforgeeks.org/ml-linear-regression/

Note that this link also contains the assumption behind linear regression and evaluation metrics which is not included in the videos.

Also, you can find implementation from scratch here: https://towardsdatascience.com/coding-linear-regression-from-scratch-c42ec079902

There are many more loss functions in addition to mean squared error. You can read about them here:
https://heartbeat.comet.ml/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

Also, you can read more about gradient descent here:
https://medium.com/geekculture/mathematics-behind-gradient-descent-f2a49a0b714f

I hope that you have gone through linear regression and gradient descent and have experimented with the hyperparameters in the provided notebook to see their effect. Now moving on further:

Normal equation is an alternative to gradient descent which is useful for small datasets:
https://youtu.be/pRSqKgwOd5k?si=fZ95wn2zx3u9LwuY
You can find its proof here:
https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/
This is an optional topic.

Moving on,
Andrew Ng playlist:
https://youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&si=b9WaafbzNVJTP2EK 
#21- #30 : Multiple Linear Regression, Feature Scaling, Polynomial Regression
Again, Andrew Ng videos provide good intuition 

Till now, we were implementing everything from scratch using numpy and pandas only. But guesswhat, python already has a library from which you can directly implement various ML models including Linear Regression model. This library is scikit-learn library, also called sklearn library. Study about it from here: 
https://www.tutorialspoint.com/scikit_learn/scikit_learn_introduction.htm
https://www.tutorialspoint.com/scikit_learn/scikit_learn_modelling_process.htm 
https://www.tutorialspoint.com/scikit_learn/scikit_learn_linear_regression.htm

Now, as told in the previous class, data preprocessing is a very crucial step in building ML models. We will cover the most important techniques in this project.

Firstly, I have told you about handling missing values using Pandas. Sklearn library also provide imputer function to handle missing data. Read about both here:
->https://www.freecodecamp.org/news/how-to-handle-missing-data-in-a-dataset/

Next, I had told about the importance of dealing with outliers and how they can hamper the results in the previous class. Study the IQR method for dealing with outliers from here:
->https://youtu.be/A3gClkblXK8?si=DWVqjzkLePYg3qf0
Now that you have known about IQR method, a box plot is a good way to observe the statistics of your data and outliers. Read about it from here:
->https://www.geeksforgeeks.org/what-is-box-plot-and-the-condition-of-outliers/

Next, I had told about the importance of selecting right features, choosing features that are actually affecting your output and avoiding choosing two features which are highly corelated. A correlation matrix is a good way to observe how two features are co-related with each other and to the output. Study about it from here:
->https://youtu.be/1fFVt4tQjRE?si=V12tPp0Bs2jUOyjJ
->https://www.geeksforgeeks.org/exploring-correlation-in-python/

Also, many a times, we need to convert categorical data into numerical values for some analysis. One-hot encoding is mostly used for this. Read about it from here:
->https://www.educative.io/blog/one-hot-encoding

Lastly, sometimes the values of one features are of much different order than of another feature leading to inconsistency while training data and hence poor model performance. Feature scaling & normalisation methods are used to solve this problem. Study about these techniques here:
->https://www.geeksforgeeks.org/ml-feature-scaling-part-2/

I am again stressing that these are the techniques which are mostly used and there are many more techniques which can be used for different cases. I would like you to explore them yourselves and google up stuff after you understand these things.

After going through these topics, you can get a summary and join all pieces together in this notebook where they have trained, both, a linear regression model with single feature and that with multiple features using scikit-learn library after doing some data pre-processing:
https://stackabuse.com/linear-regression-in-python-with-scikit-learn/

Now we will begin with classification algorithms. This week, we will study logistic regression which is the most used classification algorithm and K-nearest neighbours algorithm which is one of the easiest supervised ML algorithms. Additionally, we will also study the problem of overfitting while dealing with regressions and classification tasks and regularisation technique to resolve it.

## Week-3 resources part-1

Classification : https://www.javatpoint.com/classification-algorithm-in-machine-learning (Till Types of ML Classification Algorithms)

Logistic Regression:

- Andrew Ng playlist : https://youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&si=b9WaafbzNVJTP2EK
#31-#36

- Detailed explanation with intuition of why we are doing a particular thing: https://philippmuens.com/logistic-regression-from-scratch

- Summary:
https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148

- Implementation from scratch:
https://pub.towardsai.net/logistic-regression-from-scratch-with-only-python-code-9d3ae607e739

- Implementation using sklearn library:
https://www.educative.io/answers/how-to-implement-logistic-regression-using-the-scikit-learn-kit

## Week-3 Resources part-2

Overfitting and Regularisation

- Andrew Ng playlist : https://youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&si=b9WaafbzNVJTP2EK
#37-#41

- Overfitting: https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/

- Regularisation technique to prevent overfitting: https://www.datacamp.com/tutorial/towards-preventing-overfitting-regularization

Performance metrics for classification
- https://youtu.be/LbX4X71-TFI?si=kgTfnlMe-8-ngsrY
- https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide
(These material also contain performance metrics for regression but you can skip them)

K-Nearest Neighbor algorithm
- https://youtu.be/CQveSaMyEwM?si=-efMXsl6UeknpRJx
- https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
- https://www.geeksforgeeks.org/how-to-find-the-optimal-value-of-k-in-knn/

## Week-4 Resources

Classification with Decision Trees

- Basic terminology of trees: https://www.programiz.com/dsa/trees

- Decision trees classification explained:
https://www.youtube.com/watch?v=ZVR2Way4nwQ

- Entropy:
https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/

- Information gain and Gini index:
https://medium.com/analytics-steps/understanding-the-gini-index-and-information-gain-in-decision-trees-ab4720518ba8

- Implementation from scratch:
https://www.youtube.com/watch?v=sgQAhG5Q7iY
https://towardsdatascience.com/decision-tree-in-machine-learning-e380942a4c96 (read-only to get a basic idea)

- Sklearn library documentation:
https://scikit-learn.org/stable/modules/tree.html

- A brief summary:
https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052

Ensemble Learning

- Everything explained: https://neptune.ai/blog/ensemble-learning-guide

- More on bagging and boosting:
https://youtu.be/sN5ZcJLDMaE?si=IkOe83jPES8QYgBF

- Random Forests explained in more detail:
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&ab_channel=StatQuestwithJoshStarmer
