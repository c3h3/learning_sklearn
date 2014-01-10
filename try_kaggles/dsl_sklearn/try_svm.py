'''
Created on Jan 8, 2014

@author: c3h3
'''

import pandas as pd

train = pd.read_csv("train.csv", header = None)
labels = pd.read_csv("trainLabels.csv", header = None)
test = pd.read_csv("test.csv", header = None)


from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV




svm_model = SVC()
svm_model.fit(X = train.values, y = labels.values)

svm_model_pred = svm_model.predict(X = test.values)

results_df = pd.DataFrame(svm_model_pred)
results_df = results_df.reset_index()
results_df.columns = ["Id","Solution"]
results_df.Id += 1
results_df.to_csv("output.csv", index = None)



#import matplotlib.pyplot as plt
#plt.scatter(train.values[:,0],train.values[:,1],labels.values)
#plt.show()


from sklearn.pipeline import Pipeline





if __name__ == '__main__':
    pass


# http://scikit-learn.org/stable/auto_examples/grid_search_text_feature_extraction.html