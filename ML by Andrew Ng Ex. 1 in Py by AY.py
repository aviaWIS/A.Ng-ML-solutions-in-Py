# Import Data
import pandas as pd
data1 = pd.read_csv('machine-learning-ex1/ex1/ex1data1.txt', sep="," )
data1.columns = ['Population of City in 10,000s', 'Profit in $10,000s']

#Implement Gradient Descent for Linear Regerssion (y = mx + b)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot

def gradient_descent(x,y):
    m_temp = 0
    b_temp = 0
    itr = 700
    n = len(data1)
    alpha = 0.01

    for i in range(itr):
        y_pre = m_temp*x + b_temp
        fcost = lambda y,y_pre: (1/n)*sum((y-y_pre)**2)      #something here seems a bit off?
        cost = fcost(y,y_pre)                                 #Make it into one line somehow?
        dm = -(2/n)*sum(x*(y-y_pre))
        db = -(2/n)*sum(y-y_pre)
        m_temp = m_temp - alpha*dm
        b_temp = b_temp - alpha*db
    print("m {}, b {}, itr {}, cost {}".format(m_temp, b_temp, i, cost))

# Plot Prediction Function Over Data
    prex = np.linspace(0,(np.max(x)), num=1000)
    FoX = lambda prex: (m_temp)*prex + b_temp
    prey = []
    for px in prex:
        prey.append(FoX(px))
        
    plt1 = sns.scatterplot(x=data1['Population of City in 10,000s'], y=data1['Profit in $10,000s'], marker="x", color="r")
    plt1.set(ylim=(-5, None),xlim=(4, None))
    plt1.set_title("Figure 1: Scatter plot of training data")
    pyplot.plot(prex,prey)   

xs = np.array(data1[data1.columns[0]])
ys = np.array(data1[data1.columns[1]])               
gradient_descent(xs,ys)
