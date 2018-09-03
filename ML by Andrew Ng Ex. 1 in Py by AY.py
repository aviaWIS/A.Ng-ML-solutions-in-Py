
# coding: utf-8

# In[338]:


# Machine Learning Course by Andrew Ng - Programming Ex. 1 - A Python Solution by Avia Yehonadav

# Import Data
import pandas as pd
data1 = pd.read_csv('machine-learning-ex1/ex1/ex1data1.txt', sep="," )
data1.columns = ['Population of City in 10,000s', 'Profit in $10,000s']

# Part 1 - Plot Data
import seaborn as sns
plt1 = sns.scatterplot(x=pdData1['Population of City in 10,000s'], y=pdData1['Profit in $10,000s'], marker="x", color="r")
plt1.set(ylim=(-5, None),xlim=(4, None))
plt1.set_title("Figure 1: Scatter plot of training data")


# In[187]:



# import numpy as np
# npData1 = np.loadtxt('machine-learning-ex1/ex1/ex1data1.txt', delimiter= ',')
# print(npData1)
# plt.pyplot.scatter(pdData1['City Population'], pdData1['Food Truck Profit'])


# In[360]:


# Import Data
import pandas as pd
data1 = pd.read_csv('machine-learning-ex1/ex1/ex1data1.txt', sep="," )
data1.columns = ['Population of City in 10,000s', 'Profit in $10,000s']

#Implement Gradient Descent for Linear Regerssion (y = mx + b)
def gradient_descent(x,y):
    m_temp = 0
    b_temp = 0
    itr = 700
    n = len(data1)
    alpha = 0.01

    for i in range(itr):
        y_pre = m_temp*x + b_temp
        fcost = lambda y,y_pre: (1/n)*sum((y-y_pre)**2)      #something here seems a bit off?
        cost = fcost(y,y_pre)                                 #Make it into one line aomehow?
#         cost = (1/n)*sum([val**2 for val in (y-y_pre)])
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
        
    plt1 = sns.scatterplot(x=pdData1['Population of City in 10,000s'], y=pdData1['Profit in $10,000s'], marker="x", color="r")
    plt1.set(ylim=(-5, None),xlim=(4, None))
    plt1.set_title("Figure 1: Scatter plot of training data")
    pyplot.plot(prex,prey)
    
    

xs = np.array(data1[data1.columns[0]])
ys = np.array(data1[data1.columns[1]])               
gradient_descent(xs,ys)


# In[423]:


# Visualizing J(theta) - Cost Function
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def cost_function(x,y):
    m_temp = 0
    b_temp = 0
    n = len(data1)
    itr = 700
    alpha = 0.01
    cf_df = pd.DataFrame(columns = ['b','m','Cost'])

    for i in range(itr):
        y_pre = m_temp*x + b_temp
        fcost = lambda y,y_pre: (1/n)*sum((y-y_pre)**2)      #something here seems a bit off?
        cost = fcost(y,y_pre)                                 #Make it into one line aomehow?
        dm = -(2/n)*sum(x*(y-y_pre))
        db = -(2/n)*sum(y-y_pre)
        m_temp = m_temp - alpha*dm
        b_temp = b_temp - alpha*db
        cf_temp = [b_temp, m_temp, cost]
        cf_df = cf_df.append(pd.Series(cf_temp, index=['b','m','Cost']), ignore_index=True)
        
#     x = cf_df['b']
#     y = cf_df['m']
#     z = cf_df['Cost']
#     X_grid, Y_grid = np.meshgrid(x,y)
#     Z_grid = X_grid**2 + Y_grid**2
#     plt.contour(x, y, Z_grid)
     
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')

    plt.show()
   
cost_function(xs,ys)


# In[ ]:


cf_df = pd.DataFrame(columns = ['b(tetha_0)','m(tetha_1)','Cost'])
print(cf_df)

