#Import the relevant libraries
import numpy as np
import csv
import random

## Train data and Loading in data using csv and np
#path can be used as well if path is used then add to train = open(path+ /traindata.csv, 'rt')
#path = 'C:\Users\kavya\OneDrive\Desktop\python Files\Practice Files\traindata.csv'

train = open('traindata.csv', 'rt')
reader = csv.reader(train, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
train_data = np.array(x)


#classes comparisions
class1_2 = []
class2_3 = []
class1_3 = []

#Separating the given data into pairs, for comparsions, This is for the train data
for row in train_data:
    if row[4] == 'class-1':
        #Append the row to the relevant pairs
        class1_2.append(row)
        class1_3.append(row)
    if row[4] == 'class-2':
        class2_3.append(row)
        class1_2.append(row)
    if row[4] == 'class-3':
        class1_3.append(row)
        class2_3.append(row)

## Test data
# Repeat the same steps are previously but with test.data        
test = open('testdata.csv', 'rt')
reader = csv.reader(test, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
test_data = np.array(x)

class1_2_t = []
class2_3_t = []
class1_3_t = []

for row in test_data :
    if row[4] == 'class-1':
        class1_2_t.append(row)
        class1_3_t.append(row)
    if row[4] == 'class-2':
        class2_3_t.append(row)
        class1_2_t.append(row)
    if row[4] == 'class-3':
        class1_3_t.append(row)
        class2_3_t.append(row)
        

######################################################################################################################



#Question 2 and Question 3 

"""
The Binary Perceptron function class 1 or the first class is assigned as -1 and second class as 1.
The function has weights, bias or activation function in the classes pair and starting defining the pair of classes to compare,
and prdicts the accurancies for test data and train data for every instnance
"""
        
def PerceptronBinary(data, w, b,trainOrTest):
    #Split the data into two, with the features in x, and class' in y
    data = np.hsplit((np.array(data)),
                     np.array([4, 8]))
    x = data[0].astype(float)
    y = np.array(np.unique(data[1], return_inverse=True), dtype = "object")
    #Retain the names of the classes for printing later
    name1 = y[0][0]
    name2 = y[0][1]
    y = np.array(y[1])
    #Convert the 0 in y to -1
    y[y < 1] = -1
    
    #Create variables for pocket algorithm
    bestWeights = w
    bestBias = b
    bestAccurancy = 0
    
    #If function is defined as test, run 1 iteration 
    if trainOrTest != "Train":
        epochs = 1
    #If function is defined as train, run 20 iterations
    else:
        epochs = 20
        #Create variables for weights and bias
        w = [0.0, 0.0, 0.0, 0.0]
        b = 0
    
    # number of iterations
    for epoch in range(epochs):
        #Change accuracy to 0
        acc = 0
        #Join the x and y and shuffle the data, for best accurancy
        zipedList = list(zip(x, y))
        random.shuffle(zipedList)
        x, y = zip(*zipedList)
        
        #each row in x, set activation to 0
        for i in range(len(x)):
            a = 0
            #Calulate the activation for each feature in each row.
            for j in range(len(x[i])):
                a += (w[j] * x[i][j]) + b
            #If the a > 0, adjust to 1, if a < 0 then change to -1
            if a > 0 :
                a = 1
            else :
                a = -1
            #If the activation * the classification is <= 0 then update 
            # weights and bias on train dataset; otherwise, increase accuracy score by 1.
            if (a * y[i]) <= 0:
                if trainOrTest == "Train":
                    for j in range(len(w)):
                        w[j] = w[j] + (y[i] * x[i][j])
                    b += y[i]
            else:
                acc += 1
        #If the accuracy recorded is greater than the bestAccuracy recorded,
        # then update the bestAcc, and the weights and bias if train data
        if bestAccurancy < acc:
            bestAccurancy = acc
            if trainOrTest == "Train":
                bestWeights = w.copy()
                bestBias = b
                
    #Print the model accuracies for train and test models
    print(trainOrTest,"model accuracy for the", name1, "/",name2+":", ((bestAccurancy) / len(x)) * 100, "%")
    #Print how many lines were correct
    print("\tGot: ", (bestAccurancy), "/", len(y), "lines correct\n") 
    
    #If the data was training data, then return the bestWeights and bestBias
    if trainOrTest == "Train":
        return bestWeights, bestBias
    else:
        return

#########################################################################################################################################################################################
#########################################################################################################################################################################################



#Question 4 and Question 5 

"""
The Multi-Class Perceptron function utilises the 1-vs-rest algorithm, in which
the class of interest is given a 1, and the other classes are assigned -1.
This function will take in a whole dataset with three classes, weights and 
bias in an array, and a string defining whether the data is "Train" or "Test".
It will either produce an array of weights and an array of bias values (if 
defined as a train data), or apply the weights and bias of the train data to 
the test data to predict the classification of each instance.
"""
    
def PerceptronMultiClass(data,wArray,bArray,trainOrTest):
    #Split the data into two, with the features in x, and class' in y
    data = np.hsplit((np.array(data)),
                     np.array([4, 8]))
    x = data[0].astype(float)
    y = np.array(np.unique(data[1], return_inverse=True), dtype = "object")
    y = np.array(y[1])
    
    #Define coefficient for l2 regularisation
    #can change the coeff values according and find the accurancy
    #coeff = 0.01
    #coeff = 0.1
    #coeff = 10.0
    #coeff = 100.0
    #Create variables for pocket algorithm
    bestmultiWeights = []
    bestmultiBias = []
    #Create a copy of y
    z = y.copy()

    #For the number of classes in dataset
    for i in range(3):
        #Reset bestAccuracy to 0
        bestAccuracy = 0
        #If data is train, reset the weights, bias, bestW, bestB and
        #set the number of iterations to 20
        if trainOrTest == "Train":
            w = [0.0, 0.0, 0.0, 0.0]
            b = 0
            bestWeights = []
            
            bestBias = 0
            epochs = 20
        #If data is test, set the weight and bias to the relevant loop, and 
        #set the model iterations to 1
        else:
            w = wArray[i]
            b = bArray[i]
            epochs = 1
        
        #For the number of values in z
        for j in range(z.shape[0]):
            #If the number == 2, then change to 1, otherwise change to -1
            if z[j] == 2:
                y[j] = 1
            else:
                y[j] = -1
        #Add 1 to z for the next loop
        z += 1
        y = np.array(y)
         
        #For the number of iterations
        for epoch in range(epochs):
                #Change accuracy to 0
                acc = 0
                #Join together the x and y and shuffle the data
                zipedList = list(zip(x, y))
                random.shuffle(zipedList)
                x, y = zip(*zipedList)
                
                #For each row in x, set activation to 0
                for k in range(len(x)):
                    a = 0.0
                    #For each feature in each row, calculate the activation
                    for m in range(len(x[k])):
                        a += (w[m] * x[k][m]) + b
                    #If the activation * the classification is <= 0 then update 
                    # weights and bias on train dataset; otherwise, increase accuracy 
                    # score by 1.
                    if (a * y[k]) <= 0:
                        if trainOrTest == "Train":
                            for j in range(len(w)):
                                #for multiclassifier question 4
                                w[m] = w[m] + (y[k] * x[k][m])
                                #For Question -5 l2 Regulazation only uncomment it
                                #w[m] = w[m] + (y[k] * x[k][m]) - (2*coeff*w[m])
                            b += y[k]
                    else:
                        acc += 1
                #If the accuracy recorded is greater than the bestAccuracy recorded,
                # then update the bestAcc, and the weights and bias if train data        
                if bestAccuracy < acc:
                    bestAccuracy= acc
                    if trainOrTest == "Train":
                        bestWeights = w.copy()
                        bestBias = b
        #Print the model accuracies for train and test models            
        print(trainOrTest,"model accuracy for Class", 3-i, ":", round((bestAccuracy/len(x) *100), 2), "%")
        #Print how many lines were correct
        print("\tGot:", (bestAccuracy), "/", len(y), "lines correct\n") 
            
        #If the data is train, append the bestWeights and bestBias of each
        # loop of the function to bestmultiW and bestmultiB
        if trainOrTest == "Train":
            bestmultiWeights.append(bestWeights)
            bestmultiBias.append(bestBias)
        
        #Reset x and y ready for the next loop of the function
        x = data[0].astype(float)
        y = np.array(np.unique(data[1], return_inverse=True), dtype = "object")
        y = np.array(y[1])
    
    #If the data was training data, then return the bestWeights and bestBias
    if trainOrTest == "Train":
        return bestmultiWeights, bestmultiBias
    else:
        return

##############################################################################

## Runnig of the Models in Binary Perceptron ##
        
"""
Binary Perceptron
For the train model:
Change the data within the function to class1_2, class2_3 or class1_3
Keep the weights and bias as 0
For the test model:
Change the data within function to class1_2_t, class2_3_t or class1_3_t 
Change the weights and bias to w and b
When defining the train or test models, use "Train" or "Test"
"""        
#For Train model
#Save the weights and bias from the train model
#FOR CLASS 1 AND CLASS 2
w, b = PerceptronBinary(class1_2, 0, 0, "Train")
#FOR CLASS 2 AND CLASS 3
#w, b = PerceptronBinary(class2_3, 0, 0, "Train")
#FOR CLASS CLASS 1 AND CLASS 3
#w, b = PerceptronBinary(class1_3, 0, 0, "Train")

#For Test model
#Uses weights and bias saved from train model and then the tested data is verified
PerceptronBinary(class1_2_t, w, b,"Test")
#PerceptronBinary(class2_3_t, w, b,"Test")
#PerceptronBinary(class1_3_t, w, b,"Test")
print(w)



"""
Multi-Class Perceptron
For the train model:
Keep the data as train_data
Keep the weights and bias as 0
For the test model:
Keep the data as test_data
Keep the weights and bias as wArray and bArray
When defining the train or test models, use "Train" or "Test"
"""  
#Train model
#Save the weights and bias from the train model
wArray, bArray = PerceptronMultiClass(train_data, 0, 0,"Train")

#Test model
#Uses weights and bias saved from train model
PerceptronMultiClass(test_data, wArray, bArray,"Test")
