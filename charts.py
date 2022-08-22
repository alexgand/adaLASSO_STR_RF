################
# CHARTS:
################

# plot for the logistic function:

import numpy as np
import matplotlib
from matplotlib import cm
import seaborn
matplotlib.style.use('seaborn')
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib
#######################

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

#######################

# plot the simple regression tree:
from matplotlib import pyplot as plt
%matplotlib
from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
tree.plot_tree(clf)

################

# chart example b termos lineares e nao lineares

matplotlib.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

fig = plt.figure(figsize=plt.figaspect(0.50))

ax = fig.add_subplot(1, 2, 1)

for a in [0.5, 1, 2, 3]:
    plt.scatter(X[0], a*X[0])
ax.legend(['$0.5 \cdot x$', '$1 \cdot x$', '$2 \cdot x$','$3 \cdot x$','$4*x$','$5*x$'])
ax.set_xlabel('$x_1$, $x_2$ and $x_3$', fontsize=10)

ax = fig.add_subplot(1, 2, 2)

plt.scatter(X[5], (3*np.sin(1*X[5])))
plt.scatter(X[5], 3*np.exp(-X[5]**2))
plt.scatter(X[5], 1*X[5]**(1/2))

ax.legend(['$3 \cdot sin(x_4)$', '$3 \cdot e^{-x_{5}^{2}}$', '$\sqrt{x_6}$'], loc='upper right')
ax.set_xlabel('$x_4$, $x_5$ and $x_6$', fontsize=10)

#####

# Friedman termos lineares e nao lineares, versao 2:

matplotlib.style.use('seaborn')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

y_sem_erros = 10*np.sin(np.pi*X[0]*X[1])

fig = plt.figure(figsize=plt.figaspect(0.30))

# set up the axes for the first plot
ax = fig.add_subplot(1, 3, 1, projection='3d')

surf = ax.plot_trisurf(X[0], X[1], y_sem_erros, linewidth=0)
ax.zaxis.set_tick_params(labelsize=10)
ax.set_xlabel('$x_1$', fontsize=10, rotation=0)
ax.set_ylabel('$x_2$', fontsize=10, rotation = 0)
ax.zaxis.set_rotate_label(False) 
fake2Dline = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
ax.legend([fake2Dline], ['$10 sin ( \pi x_{1} x_{2} )$'], numpoints = 1, loc='upper left', markerscale=0)

ax = fig.add_subplot(1, 3, 2)

plt.scatter(X[0], 20*((X[0] - 0.5)**2) )

ax.legend(['$20 ( x_{3} - 0.5 )^2$'], loc='upper left', markerscale=0)
ax.set_xlabel('$x_3$', fontsize=10)

ax = fig.add_subplot(1, 3, 3)
for a in [1,5,10,15,20,25,30,35,40,45,50,55]:
    plt.scatter(X[3], a*X[3])
ax.legend(['$1 \cdot x$', '$5 \cdot x$', '$10 \cdot x$','$15 \cdot x$', '$20 \cdot x$', '$25 \cdot x$', '$30 \cdot x$', '$35 \cdot x$', '$40 \cdot x$', '$45 \cdot x$', '$50 \cdot x$', '$55 \cdot x$'])
ax.set_xlabel('$x_4$ and $x_5$', fontsize=10)
ymin, ymax = ax.get_ylim()

##########
# THE END
##########