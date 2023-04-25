import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("CrabAgePrediction.csv")                                     # Retrieve data from csv
sex = np.array(data['Sex'])
for i in range(0,len(sex)):
    if sex[i] == 'F':
        sex[i] = 0
    if sex[i] == 'M':
        sex[i] = 1
    if sex[i] == 'I':
        sex[i] = 2
length = np.array(data['Length'])
diameter = np.array(data['Diameter'])
height = np.array(data['Height'])
weight = np.array(data['Weight'])
shucked_weight = np.array(data['Shucked Weight'])
viscera_weight = np.array(data['Viscera Weight'])
shell_weight = np.array(data['Shell Weight'])
y = np.array(data['Age'])
n = len(sex)

sex_train, sex_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
length_train, length_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
diameter_train, diameter_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
height_train, height_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
weight_train, weight_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
shucked_weight_train, shucked_weight_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
viscera_weight_train, viscera_weight_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
shell_weight_train, shell_weight_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])
y_train, y_test = np.array([0.0 for x in range(3504)]), np.array([0.0 for x in range(3504,3893)])

for i in range (0,3504):
    sex_train[i] = sex[i]
    length_train[i] = length[i]
    diameter_train[i] = diameter[i]
    height_train[i] = height[i]
    weight_train[i] = weight[i]
    shucked_weight_train[i] = shucked_weight[i]
    viscera_weight_train[i] = viscera_weight[i]
    shell_weight_train[i] = shell_weight[i]
    y_train[i] = y[i]
for i in range(389):
    sex_test[i] = sex[3504+i]
    length_test[i] = length[3504+i]
    diameter_test[i] = diameter[3504+i]
    height_test[i] = height[3504+i]
    weight_test[i] = weight[3504+i]
    shucked_weight_test[i] = shucked_weight[3504+i]
    viscera_weight_test[i] = viscera_weight[3504+i]
    shell_weight_test[i] = shell_weight[3504+i]
    y_test[i] = y[3504+i]

def sigmoid(x):                                                               # Activation function
    f = 1.0/(1.0+np.exp(-x))
    return f

def Jfunc(y, yhat):                                                           # Cost function
    m = y.shape[1]
    sum = (y - yhat)**2
    J = np.sum(sum) * (1/m)
    return J

def MSE(yhat):                                                                # mean squared error function
    MSE = np.sum((y_test-yhat)**2)*(1/len(y_test[0]))
    return MSE

W1 = np.random.random((4,4))                                                  # initializing weights and bias
W2 = np.random.random((1,4))
b1 = np.zeros((4,1))
b2 = np.zeros((1,1))

def derivatives(W1,W2,Z1,Z2,A1,b1,b2):                                        # get derivatives
    dZ2 = Z2-y_train                                 
    dW2 = (1/len(y_train)) * (dZ2 @ A1.T)    
    db2 = (1/len(y_train)) * np.sum(dZ2) 
    dA1 = W2.T @ dZ2                              
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))      
    dW1 = (1/len(y_train)) * (dZ1 @ X.T)            
    db1 = (1/len(y_train)) * np.sum(dZ1)
    temp = []
    temp.append(dW1)
    temp.append(dW2)
    temp.append(db1)
    temp.append(db2)
    return temp

def SGD(Theta,Theta2,Theta3,Theta4,alpha,iterations,der):                     # Stochastic Gradient Descent function
    for it in range(iterations):
        Theta = Theta - alpha * der[0]
        Theta2 = Theta2 - alpha * der[1]
        Theta3 = Theta3 - alpha * der[2]
        Theta4 = Theta4 - alpha * der[3]
    return Theta,Theta2,Theta3,Theta4

def Train(x,wei1,wei2,b1,b2,iter,alpha):                                      # Train function
    C = []
    for i in range(0,iter):
        z1 = wei1 @ x + b1
        a1 = sigmoid(z1)      
        z2 = wei2 @ a1 + b2
        cost = Jfunc(y_train, z2)    
        C.append(cost)   
        D = derivatives(wei1,wei2,z1,z2,a1,b1,b2)
        wei1,wei2,b1,b2 = SGD(wei1,wei2,b1,b2,alpha,1,D)
    return wei1,wei2,b1,b2,C

def Test(x,wei1,wei2,b11,b22):                                                # Test function 
    z1 = wei1 @ x + b11
    a1 = sigmoid(z1)
    z2 = wei2 @ a1 + b22
    return z2
    
y_train = y_train.reshape((1,len(y_train)))
y_test = y_test.reshape((1,len(y_test)))
X = np.array([length_train,diameter_train,height_train,weight_train])
X_test = np.array([length_test,diameter_test,height_test,weight_test])

iteration = 5
alpha = 0.0001
w1,w2,b1,b2,C = Train(X,W1,W2,b1,b2,iteration,alpha)                          # Train model
yh = Test(X_test,w1,w2,b1,b2)                                                 # Test model
print('MSE: ' + str(MSE(yh)))

x = np.linspace(0,iteration,iteration)                                        # plot cost
plt.plot(x,C)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()