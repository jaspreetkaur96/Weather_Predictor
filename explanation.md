     The dataset can be found on https://github.com/words-sdsc/courseraDataSimulation/blob/master/course4-ML/daily_weather.csv
This follows the code with its explanation.

     import pandas as pd
      data=pd.read_csv('daily_weather.csv')
     Data
Pandas is a python library used for loading data in data frames. read_csv() is used to read the csv file into data frame. Loaded data can be viewed by printing data. 

     data.columns
Columns in the data frame can be printed/viewed using this command.

     data[data.isnull().any(axis='columns')]
This is performed to check which all rows have null values.
isnull() checks if there is a NaN in the data or not and marks true/false according to the value.
any() performs OR with axis =’columns’
data.isnull().any(axis='columns') evaluates true/false if that particular row contains NaN.
Finally, all the rows with a null value are printed.


     del data['number']
The number field was not useful for the prediction.Hence it was removed.

     imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)#to impute along columns
     df = imp.fit_transform(data)
     df = pd.DataFrame(data)
     df.columns = data.columns
     data = df
Now, here comes the task to replace NaN with some value. And Imputer serves the purpose. Firstly, its object is created. The object is trained on the data(fit) and then it transformation is applied(transform). 
Either we can use fir_transform method or fir and transform method.
Also, Imputer outputs a different array rather than computing on its argument frame.

Convert to a Classification Task.Binarize the relative_humidity_3pm to 0 or 1.
     
          clean_data = data.copy()
     clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1
     print(clean_data['high_humidity_label'])

In this step, we figure out that relative_humidity_3pm will be predicted on the basis of all 9am sensor readings. In order to classify, the data had to be in discrete classes. Hence, readings above 24.99 is considered as 1 else 0. 

     y=clean_data[['high_humidity_label']].copy()
     Y
The output variable is stored in y.
    
     morning_features = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am','max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',        'rain_duration_9am']
     X = clean_data[morning_features].copy()

Now the task was to divide the whole dataset into X(input) and y(output). X will contain all the features involved in the weather prediction and y contains the label that has to be predicted. 

     from sklearn.model_selection import train_test_split (X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=342)

This step involves division of the data into training and testing sets. This means that both the X and y are divided as follows:-
X-> X_train and X_test
y->y_train and y_test
 We generally keep the proportion as 75(train):25(test). 
Random sate can be any integer and is generally fixed for debugging the model.


     from sklearn.tree import DecisionTreeClassifier
     from sklearn.metrics import accuracy_score
     humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
We have used Decision Tree Classifier to train our model. As the name suggests, just like a tree it contains a root node, intermediate nodes and the leaf node. Root node and the intermediate nodes contain attributes/features and the leaf node a decision/result. The most important attribute is placed at the root node. The Decision Tree Classifier works similar to the decision taking process of humans. 

     humidity_classifier.fit(X_train, y_train)
This is where we train model on the X_train and y_train data frames. 

     predictions = humidity_classifier.predict(X_test)
This is why we have been doing all the work. Here, predictions are made on the X_test data set.

     accuracy_score(y_true = y_test, y_pred = predictions)
Here, we test the accuracy of our model by matching our predicted output and the actual output.

 We can play with hyper parameters and make our algorithm to perform better.

Congratulations, You’ve made your first machine learning model!

Assignment:- Train your model on Linear and Logistic Regression and compare your accuracies for the 3 models.


