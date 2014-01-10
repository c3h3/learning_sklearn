'''
Created on Jan 8, 2014

@author: c3h3
'''

import pandas as pd

train = pd.read_csv("train.csv", header = None)
labels = pd.read_csv("trainLabels.csv", header = None)
test = pd.read_csv("test.csv", header = None)


from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X = train.values, y = labels.values)

svm_model_pred = svm_model.predict(X = test.values)


results_df = pd.DataFrame(svm_model_pred)
results_df = results_df.reset_index()
results_df.columns = ["Id","Solution"]
results_df.Id += 1
results_df.to_csv("output.csv", index = None)





if __name__ == '__main__':
    pass