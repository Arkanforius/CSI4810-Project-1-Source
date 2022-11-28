import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plotter


#these three lines of code are meant to be edited manually before a run and choose which version of the
#dataset is being used, as well as what model type is being tested
normalize = True
compacted = True
type = "MLP"


file = "train.csv"

if compacted:
    file = "trainCompacted.csv"



rawdata = pd.read_csv(file)

def dataparse(raw):
    labels = raw["Cover_Type"]
    data = raw.drop(columns=["Id", "Cover_Type"], inplace=False)
    return data, labels
datad, datal = dataparse(rawdata)

#the following stack overflow question was used for syntax assistance
#https://stackoverflow.com/questions/19482970/get-a-list-from-pandas-dataframe-column-headers

headers = datad.columns.tolist()

if normalize:
    for h in headers:
        if datad[h].max() > 0:
            datad[h] = datad[h]/ datad[h].max()


traind, testd, trainl, testl = train_test_split(datad, datal, random_state=17)

model = ""

if type == "KNN":
    modelT = KNeighborsClassifier(1)
elif type == "NB":
    modelT = GaussianNB()
else:
    modelT = MLPClassifier(hidden_layer_sizes=(300, 150, 50), activation="tanh", verbose=True, max_iter= 600, random_state=17, solver="adam")

model = modelT.fit(traind, trainl)



results = model.predict(testd)

matrix = confusion_matrix(testl, results)
display = ConfusionMatrixDisplay(confusion_matrix=matrix)

results = cross_validate(modelT, datad, datal, scoring=("accuracy", "f1_weighted"))

print("Average of F1 score weighted averages is " + str(results["test_f1_weighted"].mean()) + " with a standard deviation of " + str(results["test_f1_weighted"].std()))
print("Average accuracy is " + str(results["test_accuracy"].mean()) + " with a standard deviation of " + str(results["test_accuracy"].std()))

display.plot()
plotter.show()

print("done")


