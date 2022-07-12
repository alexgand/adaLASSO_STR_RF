################
# CHARTS:
################

#################################
# plot for the logistic function:
#################################

import numpy as np
import matplotlib
from matplotlib import cm
import seaborn
matplotlib.style.use('seaborn')
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib

xmin = -22
xmax = 22
x = np.linspace(xmin,xmax,num=300)
fig = plt.figure(figsize=plt.figaspect(0.5))
gammas = [0.15,0.25,0.5,0.75,1,5]
for c in [-10, 10]:
    color = iter(cm.coolwarm(np.linspace(0, 1, len(gammas))))
    for gamma in gammas:
        logit = 1/(1 + np.exp(-gamma*(x-c)))
        plt.plot(x,logit,color=next(color))
plt.xlabel('location parameter $c$', fontsize=15)
plt.ylabel('$L (.)$', fontsize=15)
plt.legend(['$\gamma = 0.15$', '$\gamma = 0.25$', '$\gamma = 0.50$', '$\gamma = 0.75$', '$\gamma = 1.00$', '$\gamma = 5.00$'])


##################################
# plot the simple regression tree:
##################################

from matplotlib import pyplot as plt
%matplotlib
from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
tree.plot_tree(clf)


######################
# Plots for example a:
######################

# Figure 1:
###########

y_sem_erros =  X[0] + np.cos(np.pi * X[1]) # sem erros, soh para o grafico do paper.

matplotlib.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_title("DGP without $\epsilon$")
ax.view_init(20, -170) # https://stackoverflow.com/questions/47610614/get-viewing-camera-angles-in-matplotlib-3d-plot
# plot a 3D surface like in the example mplot3d/surface3d_demo
surf = ax.plot_trisurf(X[0], X[1], y_sem_erros, linewidth=0)
ax.zaxis.set_tick_params(labelsize=10)
ax.set_xlabel('$x_0$', fontsize=10, rotation=0)
ax.set_ylabel('$x_1$', fontsize=10, rotation = 0)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$y$', fontsize=10, rotation = 0)
# ==============
# Second subplot
# ==============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title("DGP with $\epsilon$")
ax.view_init(20, -170) # https://stackoverflow.com/questions/47610614/get-viewing-camera-angles-in-matplotlib-3d-plot
surf = ax.plot_trisurf(X[0], X[1], y, linewidth=0)
ax.zaxis.set_tick_params(labelsize=10)
ax.set_xlabel('$x_0$', fontsize=10, rotation=0)
ax.set_ylabel('$x_1$', fontsize=10, rotation = 0)
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('$y$', fontsize=10, rotation = 0)

# Figure 2:
###########

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.50))
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1)
plt.scatter(x=X[0], y=y)
plt.scatter(x=X_test[0], y=final_preds_adalasso_plus_STR_random_forest, c='tab:orange', marker='^')
ax.legend(['Train data points', 'Test predicted values'])
ax.set_xlabel('$x_0$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15)
ax = fig.add_subplot(1, 2, 2)
plt.scatter(x=X[1], y=y)
plt.scatter(x=X_test[1], y=final_preds_adalasso_plus_STR_random_forest, c='tab:orange', marker='^')
ax.legend(['Train data points', 'Test predicted values'])
ax.set_xlabel('$x_1$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15)

##################################################
# plot for example b, linear and non linear terms:
##################################################

matplotlib.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

fig = plt.figure(figsize=plt.figaspect(0.50))

# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1)
for a in [1, 2, 3]:
    plt.scatter(X[0], a*X[0])
ax.legend(['$1 \cdot x_i$', '$2 \cdot x_i$','$3 \cdot x_i$','$4*x_i$','$5*x_i$'])
ax.set_xlabel('$x_0$ to $x_2$', fontsize=15)

# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2)
plt.scatter(X[5], (3*np.sin(1*X[5])))
plt.scatter(X[5], 3*np.exp(-X[5]**2))
plt.scatter(X[5], 1*X[5]**(1/2))

ax.legend(['$3 \cdot sin(x_3)$', '$3 \cdot e^{-x_{4}^{2}}$', '$\sqrt{x_4}$'], loc='upper right')
ax.set_xlabel('$x_3$ to $x_5$', fontsize=15)
