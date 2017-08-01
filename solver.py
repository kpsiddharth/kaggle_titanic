from sklearn.neural_network.multilayer_perceptron import MLPClassifier
import numpy

clf = MLPClassifier(solver='lbfgs', early_stopping=True, validation_fraction=0.1, alpha=1e-5, hidden_layer_sizes=(16, 1))

def build_model(training_data, targets):
    clf.fit(training_data, targets)
    
def find_model_score(test_data, test_targets):
    score = clf.score(test_data, test_targets)
    print 'MLP Model Score = ', score
    
def predict_test_data(test_data):
    predictions = clf.predict(test_data)
    predictions = numpy.reshape(predictions, (predictions.size, 1))
    test_data = numpy.concatenate((test_data, predictions), axis = 1)
    
    return test_data