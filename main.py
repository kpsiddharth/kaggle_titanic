import pandas
from numpy import shape, float64, reshape, integer
from sklearn.preprocessing.data import OneHotEncoder
import numpy
from sklearn.preprocessing.label import LabelEncoder
from sklearn import preprocessing
import solver

TRAIN_DATASET_PATH = './data/train.csv'
TEST_DATASET_PATH = './data/test.csv'

def dump_output(test_output):
    output_dump = test_output[:, [19]]
    output_dump = numpy.reshape(output_dump, (output_dump.size, 1))
    
    # Get all test data to retrieve passenger IDs
    dataFrame = pandas.read_csv(TEST_DATASET_PATH)
    test_data_as_array = dataFrame.values
    test_data_as_array = test_data_as_array[:, 0]
    test_data_as_array = test_data_as_array.astype(float64)
    test_data_as_array = numpy.reshape(test_data_as_array, (test_data_as_array.size, 1))
    
    output_dump = numpy.concatenate((test_data_as_array, output_dump), axis = 1)
    return output_dump

def prepare_training_data():
    # Load the training set data frame
    dataFrame = pandas.read_csv(TRAIN_DATASET_PATH)
    raw_data_array = dataFrame.values
    
    # now delete the class variable and attach the one-hot encoded class variable
    # first complete all sequences of column deletions and then append the new columns
    targets = raw_data_array[:, 1]
    targets = targets.astype(float64)
    #targets = numpy.reshape(targets, (targets.size, 1))
    
    dataFrame = dataFrame.drop('Survived', axis=1)
    
    train_data = prepare_data(dataFrame)
    #print 'Training Data Dimensions = ', train_data[0, :]
    return (train_data, targets)
    
def prepare_test_data():
    dataFrame = pandas.read_csv(TEST_DATASET_PATH)
    test_data = prepare_data(dataFrame)

    df = pandas.DataFrame(test_data)
    df.to_csv("test_input_data.csv", header = True, index=False)
    
    #print 'Test Data Dimensions = ', test_data[0, :]
    return (test_data)
    
def prepare_data(dataFrame):
    # Normalising the NaN values now, before doing any processing
    # NM = 'Normalization Missing'
    # Dummy attribute values to enable normalization
    dataFrame.Embarked = dataFrame.Embarked.fillna('NA')
    dataFrame.Sex = dataFrame.Sex.fillna('NA')
    dataFrame.Age = dataFrame.Age.fillna(-1)
    dataFrame.SibSp = dataFrame.SibSp.fillna(0)
    dataFrame.Parch = dataFrame.Parch.fillna(0)
    
    # Since, we will need the mean of the Fare column,
    # collect this information before we begin setting the
    # Fare column to -1.
    mean_fare = dataFrame.Fare.mean()
    dataFrame.Fare = dataFrame.Fare.fillna(-1)

    # Numpy Raw Data array for pre-processing
    raw_data_array = dataFrame.values
    
    # Step 1 (a): Pre-processing of the 'Passenger Class'.
    # ####################################################
    pclass_vector = raw_data_array[:, 1]
    pclass_vector = reshape(pclass_vector, (pclass_vector.size, 1))
    pclass_vector = pclass_vector.astype(float64)
    
    encoder = OneHotEncoder(categorical_features='all', dtype=float64,
           handle_unknown='error', n_values='auto', sparse=True)
    
    # transformedPClassVector: This variable holds the passenger class
    # transformed data as vector.
    transformedPClassVector = encoder.fit(pclass_vector).transform(pclass_vector).toarray()
    
    print 'Passenger Class Features = ', shape(transformedPClassVector)
    
    # Step 1 (b): Pre-processing of Embarking Point.
    # ##############################################
    embarked_vector = raw_data_array[:, 10]

    label_encoder = LabelEncoder()
    
    embark_point_boarding_pointing = label_encoder.fit(embarked_vector).transform(embarked_vector)
    embark_point_boarding_pointing = numpy.reshape(embark_point_boarding_pointing, (embark_point_boarding_pointing.size, 1))

    embark_point_boarding_pointing = embark_point_boarding_pointing.astype(float64)

    # transformedEmbarkVector: This variable holds the embark station
    # transformed data as vector.
    transformedEmbarkVector = encoder.fit(embark_point_boarding_pointing).transform(embark_point_boarding_pointing).toarray()
    print 'Embarked Station Class Features = ', shape(transformedEmbarkVector)
    
    # Step 1 (c): Pre-processing of Gender.
    # #####################################
    gender_vector = raw_data_array[:, 3]

    # Reuse the label_encoder we created earlier
    # label_encoder = LabelEncoder()
    
    gender_vector = label_encoder.fit(gender_vector).transform(gender_vector)
    gender_vector = numpy.reshape(gender_vector, (gender_vector.size, 1))

    gender_vector = gender_vector.astype(float64)
    print 'Gender Class Features = ', shape(gender_vector)

    # transformedEmbarkVector: This variable holds the embark station
    # transformed data as vector.
    transformedGenderVector = encoder.fit(gender_vector).transform(gender_vector).toarray()
    
    # Step 1 (d): Creating range based features on Age ranges
    # #######################################################
    age_vector = raw_data_array[:, 4]
    new_age_vector = numpy.ones(age_vector.size)
    
    indices = numpy.where(numpy.logical_and(age_vector > -1, age_vector <= 10))
    numpy.put(new_age_vector, indices, 0)
    
    indices = numpy.where(numpy.logical_and(age_vector > 10, age_vector <= 20))
    numpy.put(new_age_vector, indices, 1)
    
    indices = numpy.where(numpy.logical_and(age_vector > 20, age_vector <= 30))
    numpy.put(new_age_vector, indices, 2)
    
    indices = numpy.where(numpy.logical_and(age_vector > 30, age_vector <= 40))
    numpy.put(new_age_vector, indices, 3)
    
    indices = numpy.where(numpy.logical_and(age_vector > 40, age_vector <= 50))
    numpy.put(new_age_vector, indices, 4)

    indices = numpy.where(age_vector > 50)
    numpy.put(new_age_vector, indices, 5)
    
    indices = numpy.where(age_vector == -1)
    numpy.put(new_age_vector, indices, 10)
    
    new_age_vector = numpy.reshape(new_age_vector, (new_age_vector.size, 1))
    transformedAgeRangeVector = encoder.fit(new_age_vector).transform(new_age_vector).toarray()
    
    print 'Age Class Features = ', shape(transformedAgeRangeVector)
    
    # Step 1 (e): Scaling on number of siblings and parents / children
    # ################################################################
    siblings_vector = raw_data_array[:, 5]
    siblings_vector = numpy.reshape(siblings_vector, (siblings_vector.size, 1))

    parch_vector = raw_data_array[:, 6]
    parch_vector = numpy.reshape(parch_vector, (parch_vector.size, 1))
    
    # Simple Mean - Std Deviation based scaling
    # TODO: Try with this as well
    # siblings_vector = preprocessing.scale(siblings_vector)
    # parch_vector = preprocessing.scale(parch_vector)
    
    # Using MinMaxScaler to scale the values of siblings 
    # between 0 and 1.
    min_max_scaler = preprocessing.MinMaxScaler()
    siblings_vector = min_max_scaler.fit_transform(siblings_vector)
    parch_vector = min_max_scaler.fit_transform(parch_vector)
    
    # Step 1 (f): Scaling fare spent. In the absence of information on
    # fare, we shall assume the fare to be mean.
    # ################################################################
    fare_vector = raw_data_array[:, 8]
    missing_fare_indices = numpy.where(fare_vector == -1)
    fare_vector[missing_fare_indices] = mean_fare
    fare_vector = numpy.reshape(fare_vector, (fare_vector.size, 1))

    fare_vector = min_max_scaler.fit_transform(fare_vector)
    
    raw_data_array = numpy.delete(raw_data_array, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
    raw_data_array = numpy.append(raw_data_array, transformedPClassVector, 1)
    raw_data_array = numpy.append(raw_data_array, transformedEmbarkVector, 1)
    raw_data_array = numpy.append(raw_data_array, transformedGenderVector, 1)
    raw_data_array = numpy.append(raw_data_array, transformedAgeRangeVector, 1)
    raw_data_array = numpy.append(raw_data_array, siblings_vector, 1)
    raw_data_array = numpy.append(raw_data_array, parch_vector, 1)
    raw_data_array = numpy.append(raw_data_array, fare_vector, 1)
    
    return (raw_data_array)

if __name__ == "__main__":
    #print ('Beginning learning for Titanic Model ..')
    train_data, targets = prepare_training_data()
    
    # Now we would like to split the training data 
    # into some part for testing data as well. 
    # Since there are 891 rows:
    # ~90 rows will be used for cross-validation
    # We may retain another ~90 rows for testing purposes only.
    test_data = train_data[800:, :]
    test_targets = targets[800:]
    
    train_data = train_data[:800, :]
    targets = targets[:800]

    solver.build_model(train_data, targets)
    solver.find_model_score(test_data, test_targets)
    
    test_data = prepare_test_data()
    test_output = solver.predict_test_data(test_data)
    
    dump_output = dump_output(test_output)
    dump_output = dump_output.astype(integer)
    df = pandas.DataFrame(dump_output)
    df.to_csv("test_outputs.csv", header = None, index=False)