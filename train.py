import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import utils

# import the data

df = utils.get_featured_data()
df = utils.generate_poly_feats(df, degree=2)

corr_mat = df.corr()['Avg']
feats = (corr_mat[abs(corr_mat)>=0.8].index)


# split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df[feats], df['next_rate'], test_size=.2)

# train a random forest and observer results
clf = LogisticRegression(C=100, penalty='l1')
start = time.time()
clf.fit(x_train, y_train)
duration = time.time() - start

print("dont training in {:.3f}s".format(duration/1000))

# make a prediction and evaluate it
y_pred = clf.predict(x_test)

# get a classification report
print("test set results")
print(classification_report(y_test, y_pred))

# let's try a cross-validation
from sklearn.model_selection import cross_val_score

results = cross_val_score(clf, x_train, y_train)

print("got cross validation mean values of {:.3f}".format(results.mean()))
