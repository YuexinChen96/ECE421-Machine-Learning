import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.nan)
def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        #print(Data[3745])
        #plt.figure()
        #plt.imshow(Data[3745])
        #plt.show()
        #print(Target)     # 0-9
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        #true =1 false=-1?
        #print(dataIndx)   #true or false
        #print(Data[dataIndx])  #Data[true]
        Data = Data[dataIndx]/255.
        #print(Data[0])
        Target = Target[dataIndx].reshape(-1, 1)
        #print(Target[10])
        #Target [size, 1]
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        #print(randIndx)
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        #print(Target)
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    loss = 0
    for i in range(0,len(y)):
        traning_data = x[i].flatten()
        loss =1/(2*len(y))*(np.dot(np.transpose(W),traning_data) + b - y[i])**2 + loss
    loss = loss + reg/2 * np.dot(np.transpose(W), W)
    return loss

def gradMSE(W, b, x, y, reg):
    grad_W = 0
    grad_b = 0
    for i in range(0,len(y)):
        traning_data = x[i].flatten()
        grad_W = (1/len(y)) * (np.dot(np.transpose(W),traning_data) + b - y[i]) * traning_data + grad_W
        grad_b = (1/len(y)) * (np.dot(np.transpose(W),traning_data) + b - y[i]) + grad_b
    grad_W = grad_W + reg * W
    return grad_W, grad_b

def batch_grad_descent(W, b, x, y, alpha, epochs, reg, error_tol):
    old_loss = 0;
    rate_losses = []
    validate_losses = []
    test_losses = []
    for i in range(0,epochs):
        new_loss = MSE(W,b,x,y,reg)
        validate_loss = MSE(W,b,validData,validTarget,reg)
        test_loss = MSE(W,b,testData,testTarget,reg)
        
        grad_W, grad_b = gradMSE(W,b,x,y,reg)
        W = W - grad_W * alpha
        b = b - grad_b * alpha
        if abs(new_loss - old_loss) < error_tol:
            final_W = W
            final_b = b
        old_loss = new_loss
        print(new_loss,validate_loss,test_loss, i)
        rate_losses.append(new_loss)
        validate_losses.append(validate_loss)
        test_losses.append(test_loss)
    return rate_losses,validate_losses,test_losses


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    pass

def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    pass

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    pass

def batch_GD(x,y):
    training = np.empty((len(y),len(trainData[0].flatten())))
    for i in range(0,len(y)):
        training[i] = x[i].flatten()
    return training

if __name__=="__main__":
    #trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    #weight = np.zeros(len(trainData[0].flatten()))
    #rate_losses,validate_losses,test_losses = batch_grad_descent(weight,0, trainData, trainTarget, 0.005, 5000,0.1,1e-7)
    #print(rate_losses,validate_losses,test_losses)
    #MSE(weight, 0, trainData, trainTarget, 0)
    #a = tf.Variable(tf.random.normal([2,2], 5.0, 10.0))
    #weight = np.ones(len(trainData[0].flatten()))
    #print(weight*trainData[0].flatten())
    #final = trainData * weight
    #print(trainTarget)
    #print(trainData[0].flatten())
    #print(trainData[0])
    #plt.figure()
    #plt.imshow(trainData[0])
    #plt.colorbar()
    #plt.grid(False)
    #plt.show()
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    batch_grad = batch_GD(testData,testTarget)
    print(batch_grad)