################
# This is the code needed to run the simulations on the
# "Combining LASSO-Type Methods with a Smooth Transition Random Forest" paper from Gandini and Ziegelman, 2022.
################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.robjects.vectors import DataFrame, FloatVector, IntVector, StrVector, ListVector, Matrix, FloatMatrix
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri# Defining the R script and loading the instance in Python
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
# tem que instalar previamente o glmnet com sudo apt-get install -y r-cran-glmnet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tqdm import tqdm
from sklearn import datasets
from timeit import default_timer as timer
from datetime import timedelta
from collections import OrderedDict
import matplotlib
from matplotlib import cm
from multiprocessing import Pool, cpu_count
import pickle

%matplotlib

# using R packages in python:
base = importr('base')
stats = importr('stats')
glmnet = importr("glmnet")

# function to create STR trees in parallel, using multiprocessing:
def STR_tree_parallel(qty_of_trees):

    # https://stackoverflow.com/questions/9209078/using-python-multiprocessing-with-different-random-seed-for-each-process
    # https://stackoverflow.com/questions/29854398/seeding-random-number-generators-in-parallel-programs
    np.random.RandomState()

    # bootstrap sample:
    indexes = np.random.choice(range(len(y_train)), size=len(y_train))
    X_sample = X_train.iloc[indexes,:]
    y_sample = y_train.iloc[indexes]
    residuos_sample = residuos.iloc[indexes]
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_sample_r = robjects.conversion.py2rpy(X_sample)
        y_sample_r = robjects.conversion.py2rpy(y_sample)
        residuos_sample_r = robjects.conversion.py2rpy(residuos_sample)

    if title == 'adaLASSO + STR RF':    
        # y is the residuals:
        result_r = grow_tree_f_r(X_sample_r, residuos_sample_r, p=2/3, d_max=d_max, gamma=robjects.r.seq(0.5,5,0.01),node_obs=n/200)
    elif title == 'STR RF':
        # y is the response:
        result_r = grow_tree_f_r(X_sample_r, y_sample_r, p=2/3, d_max=d_max, gamma=robjects.r.seq(0.5,5,0.01),node_obs=n/200)
    preds_2_r = predict_smooth_tree_f_r(result_r,newx=X_test_r)
    preds_2 = recurse_r_tree(preds_2_r)
    preds_2 = preds_2[:,0]
    
    return preds_2

# suppress displaying divide by zero error:
# https://stackoverflow.com/questions/31688667/how-to-suppress-the-error-message-when-dividing-0-by-0-using-np-divide-alongsid
np.seterr(all='ignore')

# converts R objects to python as appropriate:
def recurse_r_tree(data):
    r_dict_types = [DataFrame, ListVector]
    r_array_types = [FloatVector, IntVector, Matrix, FloatMatrix]
    r_list_types = [StrVector]
    if type(data) in r_dict_types:
        return OrderedDict(zip(data.names, [recurse_r_tree(elt) for elt in data]))
    elif type(data) in r_list_types:
        return [recurse_r_tree(elt) for elt in data]
    elif type(data) in r_array_types:
        return np.array(data)
    else:
        if hasattr(data, "rclass"):  # An unsupported r class
            raise KeyError('Could not proceed, type {} is not defined'
                           'to add support for this type, just add it to the imports '
                           'and to the appropriate type list above'.format(type(data)))
        else:
            return data  # We reached the end of recursion

# adaLASSO from R function:
def adalasso_from_r(X, y, ridge_1st_step=False, intercept=True):

    # #https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html
    with localconverter(robjects.default_converter + pandas2ri.converter):
        X_r = robjects.conversion.py2rpy(X)
        y_r = robjects.conversion.py2rpy(y)

    # 1st step: 
    if ridge_1st_step:
        lasso = glmnet.cv_glmnet(x=base.as_matrix(X_r), y=base.as_matrix(y_r), alpha=0, intercept=intercept)
    else:
        lasso = glmnet.cv_glmnet(x=base.as_matrix(X_r), y=base.as_matrix(y_r), alpha=1, intercept=intercept)
        # alpha=1 is lasso, alpha=0 is ridge
    coefs = stats.coef(lasso)
    posicoes = recurse_r_tree(coefs.slots['i'])
    posicoes = list(posicoes)
    dim = recurse_r_tree(coefs.slots['Dim'])[0]
    coefs = recurse_r_tree(coefs.slots['x'])
    coefs_1st_step = [0] * dim
    for pos in posicoes:
        coefs_1st_step[pos] = coefs[posicoes.index(pos)]

    weights = 1/abs(np.array(coefs_1st_step))
    weights = np.where(weights == np.inf, 999999999, weights) ## Replacing values estimated as Infinite for 999999999
    weights = pd.Series(weights)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        weights_r = robjects.conversion.py2rpy(weights)

    # 2st step:
    lasso2 = glmnet.cv_glmnet(x=base.as_matrix(X_r), y=base.as_matrix(y_r), alpha=1, intercept=intercept, penalty_factor=weights_r)
    coefs = stats.coef(lasso2)
    posicoes = recurse_r_tree(coefs.slots['i'])
    posicoes = list(posicoes)
    dim = recurse_r_tree(coefs.slots['Dim'])[0]
    coefs = recurse_r_tree(coefs.slots['x'])
    final_coefs = [0] * dim
    for pos in posicoes:
        final_coefs[pos] = coefs[posicoes.index(pos)]

    #retira o 1o item se nao for usar o intercepto:
    if intercept == False:
        final_coefs = final_coefs[1:]
    
    final_coefs = np.array(final_coefs)

    return final_coefs

#####################
# DATASET to work on:
#####################

# dataset = 'exemplo a'
# dataset = 'exemplo b'

# Synthetic datasets:
# dataset = 'Fan & Zhang #1'
# dataset = 'Fan & Zhang #2'
# dataset = 'Friedman #1'
# dataset = 'Friedman #2'
# dataset = 'Friedman #3'

# Real datasets:
# dataset = 'abalone'
# dataset = 'ailerons'
dataset = 'bodyfat'
# dataset = 'boston'
# dataset = 'california'
# dataset = 'cholesterol'
# dataset = 'debutanizer'
# dataset = 'diabetes'
# dataset = 'diamonds'
# dataset = 'liver-disorders'
# dataset = 'sulfur'

real_datasets = ['abalone', 'bodyfat', 'boston', 'california', 'debutanizer', 'ailerons', 'diabetes', 'liver-disorders', 'sulfur', 'diamonds', 'smart grid', 'cholesterol', 'air quality']
large_datasets = ['ailerons','diamonds']

#############
# PARAMETERS:
#############

intercepto_lasso = True
# max depth of a STR tree:
d_max = 4
# number of covariates:
p = 10
# number of samples:
n = 1000
if (dataset == 'exemplo a'):
    n = 3000
# number of trees in the STR RANDOM FOREST and BooST models:
number_of_trees = 250
# number of repetitions of: generating data (for syntetic datasets) and training.
repetitions = 500
# all of them:
models_to_train = ['adaLASSO', 'adaLASSO + STR RF', 'adaLASSO + RF', 'BooST', 'RF', 'STR RF', 'SVR', 'OLS']
# debug p/ ganhar tempo:
# models_to_train = ['adaLASSO', 'adaLASSO + STR RF', 'adaLASSO + RF', 'RF', 'SVR', 'OLS']

# analysis of predictive performance by the number of trees in the STR RANDOM FOREST:
# number_of_trees_analysis = True
number_of_trees_analysis = False
if number_of_trees_analysis:
    repetitions = 500
    n = 1000
    p = 10
    dataset = 'exemplo b'

#############

# medir tempo: https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
start = timer()

np.random.seed(0)

def generate_data(dataset, p=p, n=n):

    flag_return_coefs = False

    if dataset == 'boston':
        # get data:
        boston = datasets.load_boston()
        X = pd.DataFrame(boston['data'], columns=boston['feature_names'])
        y = boston['target']
        y = pd.Series(y)

    elif dataset == 'diabetes':

        diabetes = datasets.load_diabetes()
        X = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
        y = diabetes['target']
        y = pd.Series(y)

    elif dataset == 'california':
        X, y = datasets.fetch_california_housing(return_X_y=True)
        X = pd.DataFrame(X)
        y = pd.Series(y)

    elif dataset == 'liver-disorders':
        data = datasets.fetch_openml(name='liver-disorders')
        X = data['data']
        y = data['target']

    elif dataset == 'ailerons':
        data = datasets.fetch_openml(name='ailerons')
        X = data['data']
        y = data['target']

    elif dataset == 'sulfur':
        data = datasets.fetch_openml(name='sulfur')
        X = data['data']
        y = data['target']

    elif dataset == 'debutanizer':
        data = datasets.fetch_openml(name='debutanizer')
        X = data['data']
        y = data['target']

    elif dataset == 'cholesterol':
        data = datasets.fetch_openml(name='cholesterol')
        X = data['data']
        X = X[['age','trestbps','thalach','oldpeak']]
        y = data['target']

    elif dataset == 'bodyfat':
        data = datasets.fetch_openml(name='bodyfat')
        X = data['data']
        y = data['target']

    elif dataset == 'diamonds':
        # https://www.kaggle.com/datasets/shivam2503/diamonds
        data = pd.read_csv('/home/alega/dissertacao/data/diamonds.csv')
        y = data['price']
        X = data[['carat','depth','table','x','y','z']]

    elif dataset == 'abalone':
        # https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
        # https://archive.ics.uci.edu/ml/datasets/abalone
        cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
        data = pd.read_csv('/home/alega/dissertacao/data/abalone.data', names=cols)
        y = data['Rings']
        data = data[[col for col in data if ((col != 'Sex') and (col != 'Rings') )]]
        X = data.copy()

    elif dataset == 'air quality':
        # https://archive.ics.uci.edu/ml/datasets/air+quality
        data = pd.read_csv('/home/alega/dissertacao/data/AirQualityUCI.csv', delimiter=';')
        data = data.replace(-200, np.nan) # nans estao como -200 no dataset
        data = data.dropna()

        data = data[[col for col in data if not col.startswith('Unn')]]
        data = data[[col for col in data if ((col != 'Date') and (col != 'Time') )]]
        data = data.dropna()
        y = data['Rings']
        data = data[[col for col in data if ((col != 'Sex') and (col != 'Rings') )]]
        X = data.copy()

    elif dataset == 'smart grid':
        # https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability
        data = pd.read_csv('/home/alega/dissertacao/data/smart_grid_stability_augmented.csv')
        y = data['stab']
        X = data[[col for col in data if not col.startswith('stab')]]

    elif dataset == 'exemplo b':

        media_padrao = np.pi/2
        correlacao_padrao = 0.85
        std_dev_padrao = 0.5

        media_do_erro = 0
        std_dev_do_erro = 0.25

        mean = np.array([media_padrao] * p)

        corr = np.identity(n=p)
        corr = np.where(corr == 1, corr, correlacao_padrao) # matrix
        corr = pd.DataFrame(corr)
        
        std_devs= pd.Series(np.array([std_dev_padrao] * p))
        # https://stackoverflow.com/questions/30251737/how-do-i-convert-list-of-correlations-to-covariance-matrix
        cov = corr.multiply(std_devs.multiply(std_devs.T.values))

        data = np.random.multivariate_normal(mean, cov, size=n, check_valid='warn', tol=1e-8)
        data = pd.DataFrame(data)

        # forca nao ter valor negativo (duas rodadas), sorteia de novo o particular valor negativo:
        data = pd.DataFrame(np.where(data < 0, np.random.normal(loc=media_padrao, scale=std_dev_padrao), data))
        data = pd.DataFrame(np.where(data < 0, np.random.normal(loc=media_padrao, scale=std_dev_padrao), data))

        data = pd.DataFrame(np.where(data > np.pi, np.random.normal(loc=media_padrao, scale=std_dev_padrao), data))
        data = pd.DataFrame(np.where(data > np.pi, np.random.normal(loc=media_padrao, scale=std_dev_padrao), data))

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)

        a = np.random.randint(1,4)
        b = np.random.randint(1,4)
        c = np.random.randint(1,4)

        if number_of_trees_analysis:
            a = b = c = 1

        y = a*data[0] + b*data[1] + c*data[2] + 3*np.sin(1*data[3]) + 3*np.exp(-data[5]**2) + data[5]**(1/2) + erro 

        y = y.dropna()
        data = data.reindex(y.index)
        X = pd.DataFrame(data.values)
        y = y.values
        y = pd.Series(y)

        flag_return_coefs = True

        sum_lin_coefs = a+b+c

    elif dataset == 'exemplo a':

        p = 2

        media_padrao = 0 
        std_dev_padrao = 0.75
        correlacao_padrao = 0.85
        
        media_do_erro = 0
        std_dev_do_erro = 0.75

        mean = np.array([media_padrao] * p)

        corr = np.identity(n=p)
        corr = np.where(corr == 1, corr, correlacao_padrao) # matrix
        corr = pd.DataFrame(corr)

        std_devs= pd.Series(np.array([std_dev_padrao] * p))
        # https://stackoverflow.com/questions/30251737/how-do-i-convert-list-of-correlations-to-covariance-matrix
        cov = corr.multiply(std_devs.multiply(std_devs.T.values))

        data = np.random.multivariate_normal(mean, cov, size=n, check_valid='warn', tol=1e-8)
        data = pd.DataFrame(data)

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)

        y = data[0] + np.cos(np.pi * data[1]) + erro

        y = y.dropna()
        data = data.reindex(y.index)
        X = pd.DataFrame(data.values)
        y = y.values
        y = pd.Series(y)

    elif dataset == 'Fan & Zhang #1': 
        
        p = 2

        media_padrao = 0
        correlacao_padrao = 2**(-1/2)
        std_dev_padrao = 1
        media_do_erro = 0
        std_dev_do_erro = 1

        mean = np.array([media_padrao] * p)

        corr = np.identity(n=p)
        corr = np.where(corr == 1, corr, correlacao_padrao) # matrix
        corr = pd.DataFrame(corr)
        
        std_devs= pd.Series(np.array([std_dev_padrao] * p))
        # https://stackoverflow.com/questions/30251737/how-do-i-convert-list-of-correlations-to-covariance-matrix
        cov = corr.multiply(std_devs.multiply(std_devs.T.values))

        data = np.random.multivariate_normal(mean, cov, size=n, check_valid='warn', tol=1e-8)
        data = pd.DataFrame(data)

        x1 = pd.Series(np.random.uniform(low=0.0, high=1.0, size=n))
        x2 = data[0]
        x3 = data[1]

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)

        y = (np.sin(60*x1))*x2 + 4*x1*(1 - x1)*x3 + erro

        data = pd.DataFrame([x1,x2,x3]).T
        data.columns=['x1','x2','x3']
        X = data

    elif dataset == 'Fan & Zhang #2': 
        
        p = 2

        media_padrao = 0
        correlacao_padrao = 2**(-1/2)
        std_dev_padrao = 1
        media_do_erro = 0
        std_dev_do_erro = 1

        mean = np.array([media_padrao] * p)

        corr = np.identity(n=p)
        corr = np.where(corr == 1, corr, correlacao_padrao) # matrix
        corr = pd.DataFrame(corr)
        
        std_devs= pd.Series(np.array([std_dev_padrao] * p))
        # https://stackoverflow.com/questions/30251737/how-do-i-convert-list-of-correlations-to-covariance-matrix
        cov = corr.multiply(std_devs.multiply(std_devs.T.values))

        data = np.random.multivariate_normal(mean, cov, size=n, check_valid='warn', tol=1e-8)
        data = pd.DataFrame(data)

        x1 = pd.Series(np.random.uniform(low=0.0, high=1.0, size=n))
        x2 = data[0]
        x3 = data[1]

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)

        y = (np.sin(8*np.pi*(x1-0.5)))*x2 + ( np.exp(-((4*x1 - 1)**2) ) + np.exp(-((4*x1 - 3)**2) ) - 1.4 ) * x3 + erro

        data = pd.DataFrame([x1,x2,x3]).T
        data.columns=['x1','x2','x3']
        X = data

    elif dataset == 'Friedman #1': 

        n = 1000
        p = 10
        media_do_erro = 0
        std_dev_do_erro = 1

        data = pd.DataFrame()
        for col in range(p):
            data[col] = np.random.uniform(low=0.0, high=1.0, size=n)

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)
        y = 10*np.sin(np.pi*data[0]*data[1]) + 20*((data[2] - 0.5)**2) + 10*data[3] + 5*data[4] + erro

        X = data

    elif dataset == 'Friedman #2': 

        n = 1000
        p = 4
        media_do_erro = 0
        std_dev_do_erro = 1

        data = pd.DataFrame()

        data[0] = np.random.uniform(low=0.0, high=100.0, size=n)
        data[1] = np.random.uniform(low=(20*2*np.pi), high=(280*2*np.pi), size=n)
        data[2] = np.random.uniform(low=(0.0), high=(1.0), size=n)
        data[3] = np.random.uniform(low=(1.0), high=(11.0), size=n)

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)
        y = (data[0]**2 + (data[1]*data[2] - (1/(data[1]*data[3])))**2 )**(1/2) + erro

        X = data

    elif dataset == 'Friedman #3': 

        n = 1000
        p = 4
        media_do_erro = 0
        std_dev_do_erro = 1

        data = pd.DataFrame()

        data[0] = np.random.uniform(low=0.0, high=100.0, size=n)
        data[1] = np.random.uniform(low=(20*2*np.pi), high=(280*2*np.pi), size=n)
        data[2] = np.random.uniform(low=(0.0), high=(1.0), size=n)
        data[3] = np.random.uniform(low=(1.0), high=(11.0), size=n)

        erro = np.random.normal(loc=media_do_erro, scale=std_dev_do_erro, size=n)
        y = np.arctan( ((data[1]*data[2]) - (1/(data[1]*data[3]) )) /data[0] ) + erro

        X = data

    if flag_return_coefs:
        return X, y, sum_lin_coefs
    else:
        return X, y

# r source file:
r = robjects.r
r['source']('/home/alega/dissertacao/smooth_tree.r')# Loading the function we have defined in R.
# r source functions:
grow_tree_f_r = robjects.globalenv['grow_tree']
predict_smooth_tree_f_r = robjects.globalenv['predict.SmoothTree']
boost_f_r = robjects.globalenv['BooST']
predict_boost_f_r = robjects.globalenv['predict.BooST']

print('')
print('Dataset name:', dataset)
print('')

# store results:
resultados = []

print('Training models:')

if number_of_trees_analysis:

    print('\nStarting NUMBER OF TREES analysis with dataset:', dataset, '\n')

    number_of_trees_inicial = 75
    number_of_trees_final = 500
    step = 25
    primeiras = [1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,50]

    for qty_of_trees_analysis in tqdm(primeiras + list(range(number_of_trees_inicial, number_of_trees_final, step)) + [number_of_trees_final]):

        print('qty of trees:', qty_of_trees_analysis)

        for _ in tqdm(range(repetitions)):

            data, y, sum_lin_coefs = generate_data(dataset)
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

            # train/test split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

            #https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html
            with localconverter(robjects.default_converter + pandas2ri.converter):
                X_train_r = robjects.conversion.py2rpy(X_train)
                y_train_r = robjects.conversion.py2rpy(y_train)
                X_test_r = robjects.conversion.py2rpy(X_test)
                y_test_r = robjects.conversion.py2rpy(y_test)


            title = 'adaLASSO'
            if title in models_to_train:
                start_model = timer()
                coefs_adalasso = adalasso_from_r(X_train, y_train, ridge_1st_step=False, intercept=intercepto_lasso)
                X_test_new = X_test.copy()
                if intercepto_lasso == True:
                    X_test_new.insert(0, 'intercept', 1) # intercepto
                preds_adalasso = (X_test_new * coefs_adalasso).sum(axis=1) # usado para outros modelos daqui pra frente.
                preds = preds_adalasso
                MSE = mean_squared_error(y_true=y_test,y_pred=preds)
                relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
                # residuos do treinamento adalasso:
                X_train_new = X_train.copy()
                if intercepto_lasso == True:
                    X_train_new.insert(0, 'intercept', 1) # intercepto
                preds_train = (X_train_new * coefs_adalasso).sum(axis=1)
                residuo = y_train - preds_train
                residuos = pd.Series(residuo) # usado para outros modelos daqui pra frente.
                end_model = timer()

            title = 'adaLASSO + STR RF'
            if title in models_to_train:
                start_model=timer()

                # geracao das predictions das STR trees em paralelo:
                # quantity of CPUs for multiprocessing:
                if qty_of_trees_analysis < int(cpu_count()-1):
                    qtd_CPUs = qty_of_trees_analysis
                else:
                    qtd_CPUs = int(cpu_count()-1)
                pool = Pool(qtd_CPUs) # number of CPUs
                #paralell processing:
                results = pool.map(STR_tree_parallel, [1] * qty_of_trees_analysis)
                preds = pd.DataFrame(results).mean()
                
                preds = preds_adalasso + preds.values
                final_preds_adalasso_plus_STR_random_forest = preds.copy() # para graficos
                MSE = mean_squared_error(y_true=y_test, y_pred=preds)
                relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
                end_model = timer()
                if dataset == 'exemplo b':
                    resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs, qty_of_trees_analysis])
                else:
                    resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        # mostra resultados parciais:
        resultados_df = pd.DataFrame(resultados, columns=['modelo', 'MSE', 'relative error', 'run time', 'sum lin coefs', 'number of trees'])
        resultados_df = resultados_df[resultados_df['modelo'] == 'adaLASSO + STR RF']
        resultados_df = resultados_df[[col for col in resultados_df.columns if col != 'modelo']]
        resultados_df = resultados_df.groupby('number of trees').mean()
        print(resultados_df)

else:

    # gera uma vez soh:
    if dataset in real_datasets:
        data, y = generate_data(dataset)
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    for _ in tqdm(range(repetitions)):

        # limita tamanho do dataset para testes (mais rapido), faz sample every repetition:
        if dataset in large_datasets:
            qtd_obs_a_manter = 5000
            data = data.sample(qtd_obs_a_manter).sort_index()
            y = y.loc[data.index]
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

        # gera banco de dados sintetico a cada repeticao:
        elif dataset not in real_datasets:
            if dataset == 'exemplo b':
                data, y, sum_lin_coefs = generate_data(dataset)
            else:
                data, y = generate_data(dataset)
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

        # train/test split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        #https://rpy2.github.io/doc/v3.0.x/html/generated_rst/pandas.html
        with localconverter(robjects.default_converter + pandas2ri.converter):
            X_train_r = robjects.conversion.py2rpy(X_train)
            y_train_r = robjects.conversion.py2rpy(y_train)
            X_test_r = robjects.conversion.py2rpy(X_test)
            y_test_r = robjects.conversion.py2rpy(y_test)

        #########
        # MODELS:
        #########

        title = 'adaLASSO'
        if title in models_to_train:
            start_model = timer()
            coefs_adalasso = adalasso_from_r(X_train, y_train, ridge_1st_step=False, intercept=intercepto_lasso)
            X_test_new = X_test.copy()
            if intercepto_lasso == True:
                X_test_new.insert(0, 'intercept', 1) # intercepto
            preds_adalasso = (X_test_new * coefs_adalasso).sum(axis=1) # usado para outros modelos daqui pra frente.
            preds = preds_adalasso
            MSE = mean_squared_error(y_true=y_test,y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            # residuos do treinamento adalasso:
            X_train_new = X_train.copy()
            if intercepto_lasso == True:
                X_train_new.insert(0, 'intercept', 1) # intercepto
            preds_train = (X_train_new * coefs_adalasso).sum(axis=1)
            residuo = y_train - preds_train
            residuos = pd.Series(residuo) # usado para outros modelos daqui pra frente.
            end_model = timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'STR RF'
        if title in models_to_train:
            start_model=timer()

            # geracao das predictions das STR trees em paralelo:
            # quantity of CPUs for multiprocessing:
            if number_of_trees < int(cpu_count()-1):
                qtd_CPUs = number_of_trees
            else:
                qtd_CPUs = int(cpu_count()-1)
            pool = Pool(qtd_CPUs) # number of CPUs
            #paralell processing:
            results = pool.map(STR_tree_parallel, [1] * number_of_trees)
            preds = pd.DataFrame(results).mean()

            MSE = mean_squared_error(y_true=y_test, y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model = timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'adaLASSO + STR RF'
        if title in models_to_train:
            start_model=timer()

            # geracao das predictions das STR trees em paralelo:
            # quantity of CPUs for multiprocessing:
            if number_of_trees < int(cpu_count()-1):
                qtd_CPUs = number_of_trees
            else:
                qtd_CPUs = int(cpu_count()-1)
            pool = Pool(qtd_CPUs) # number of CPUs
            #paralell processing:
            results = pool.map(STR_tree_parallel, [1] * number_of_trees)
            preds = pd.DataFrame(results).mean()

            preds = preds_adalasso + preds.values
            final_preds_adalasso_plus_STR_random_forest = preds.copy() # para graficos
            MSE = mean_squared_error(y_true=y_test, y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model = timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'adaLASSO + RF'
        if title in models_to_train:
            start_model=timer()
            model = RandomForestRegressor()
            model.fit(X_train, residuos)
            preds_RF = model.predict(X_test)
            preds = preds_adalasso + preds_RF
            MSE = mean_squared_error(y_true=y_test, y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model = timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'RF'
        if title in models_to_train:
            start_model=timer()
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            MSE = mean_squared_error(y_true=y_test, y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model=timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'SVR'
        if title in models_to_train:
            start_model=timer()
            model = SVR()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            MSE = mean_squared_error(y_true=y_test, y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model=timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'OLS'
        if title in models_to_train:
            start_model=timer()
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            MSE = mean_squared_error(y_true=y_test, y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model=timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        title = 'BooST'
        if title in models_to_train:
            start_model=timer()
            result_r = boost_f_r(X_train_r, y_train_r, v=0.2, p=2/3, d_max=d_max, gamma=robjects.r.seq(0.5,5,0.01), M=number_of_trees, display=False,stochastic=False,s_prop=0.5, node_obs=n/200, random=False)
            preds_r = predict_boost_f_r(result_r,newx=X_test_r)
            preds = recurse_r_tree(preds_r)
            MSE = mean_squared_error(y_true=y_test,y_pred=preds)
            relative_error = ((((preds-y_test)/y_test)**2).sum()/len(y_test))**(1/2)
            end_model=timer()
            if dataset == 'exemplo b':
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model), sum_lin_coefs])
            else:
                resultados.append([title, MSE, relative_error, timedelta(seconds=end_model-start_model)])

        # mostra resultados parciais:        
        if dataset == 'exemplo b':
            resultados_df = pd.DataFrame(resultados, columns=['modelo', 'MSE', 'relative error', 'run time', 'sum lin coefs'])
            resultados_df['run time'] = resultados_df['run time'].apply(lambda x: x.total_seconds()) # run time for each repetition in seconds
            resultados_df = resultados_df.sort_values('MSE')
            resultados_df = resultados_df.groupby(['sum lin coefs', 'modelo']).mean()
            resultados_df = resultados_df[['relative error']]
            # https://stackoverflow.com/questions/52566616/transposing-selected-multiindex-levels-in-pandas-dataframe
            resultados_df = resultados_df.stack().unstack(level=1)
            resultados_df = resultados_df.sort_index()
            resultados_df = resultados_df.droplevel(1)
            print(resultados_df)
        
        else:
            print('')
            print('\nDATASET:', dataset)
            print('X.shape:', X.shape)
            resultados_df = pd.DataFrame(resultados, columns=['modelo', 'MSE', 'relative error', 'run time'])
            resultados_df['run time'] = resultados_df['run time'].apply(lambda x: x.total_seconds()) # run time for each repetition in seconds
            resultados_df = resultados_df.groupby('modelo').mean()
            # transforma em RMSE:
            resultados_df['RMSE'] = resultados_df['MSE']**(1/2)
            resultados_df = resultados_df.sort_values('MSE')
            resultados_df = resultados_df[['RMSE', 'relative error', 'run time']]
            print(resultados_df)

############
# RESULTS:
############

if number_of_trees_analysis:

    resultados_df = pd.DataFrame(resultados, columns=['modelo', 'MSE', 'relative error','run time', 'sum lin coefs', 'number of trees'])
    resultados_df = resultados_df[[col for col in resultados_df.columns if col != 'modelo']]
    resultados_df = resultados_df.groupby('number of trees').mean()

    matplotlib.style.use('seaborn')
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    fig = plt.figure(figsize=plt.figaspect(0.50))

    ax = fig.add_subplot(1, 2, 1)

    resultados_df['number of trees'] = resultados_df.index
    plt.scatter(x=resultados_df['number of trees'], y=resultados_df['MSE'], marker='o')

    from scipy.interpolate import make_interp_spline, BSpline
    xnew = np.linspace(resultados_df.index.min(), resultados_df.index.max(), 300)  
    spl = make_interp_spline(resultados_df.index, resultados_df['MSE'], k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth)

    ax.set_xlabel('Number of trees', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)

    ax = fig.add_subplot(1, 2, 2)

    plt.scatter(x=resultados_df['number of trees'], y=np.log(resultados_df['MSE']), marker='o')

    xnew = np.linspace(resultados_df.index.min(), resultados_df.index.max(), 300)  
    spl = make_interp_spline(resultados_df.index, np.log(resultados_df['MSE']), k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth)

    ax.set_xlabel('Number of trees', fontsize=12)
    ax.set_ylabel('log (MSE)', fontsize=12)

elif dataset == 'exemplo b':

    resultados_df = pd.DataFrame(resultados, columns=['modelo', 'MSE', 'relative error', 'run time', 'sum lin coefs'])
    resultados_df['run time'] = resultados_df['run time'].apply(lambda x: x.total_seconds()) # run time for each repetition in seconds

    print('Quantas vezes cada sum lin coefs rodou:', resultados_df.groupby(['sum lin coefs']).count() / len(np.unique(resultados_df['modelo'])))
    print('Total runs:', (resultados_df.groupby(['sum lin coefs']).count() / len(np.unique(resultados_df['modelo']))).sum())

    run_time =  resultados_df.groupby('modelo').mean()
    print('Run time:', run_time)
    print('BooST is', run_time.loc['BooST', 'run time'] / run_time.loc['adaLASSO + STR RF', 'run time'], 'times slower than adalasso + STR RF')

    resultados_df = resultados_df.sort_values('MSE')
    resultados_df = resultados_df.groupby(['sum lin coefs', 'modelo']).mean()
    resultados_df = resultados_df[['relative error']]
    # https://stackoverflow.com/questions/52566616/transposing-selected-multiindex-levels-in-pandas-dataframe
    resultados_df = resultados_df.stack().unstack(level=1)
    resultados_df = resultados_df.sort_index()
    resultados_df = resultados_df.droplevel(1)
    print(resultados_df)
    end = timer()
    print('Tempo total para análise:', timedelta(seconds=end-start))

    # formatacao para a dissertacao:
    resultados_df = resultados_df[['OLS', 'adaLASSO', 'SVR', 'RF', 'STR RF', 'BooST', 'adaLASSO + RF', 'adaLASSO + STR RF']] # reordena as colunas.
    print(resultados_df.round(4).to_latex())

else:

    resultados_df = pd.DataFrame(resultados, columns=['modelo', 'MSE', 'relative error', 'run time'])
    resultados_df['run time'] = resultados_df['run time'].apply(lambda x: x.total_seconds()) # run time for each repetition in seconds

    resultados_df = resultados_df.groupby('modelo').mean()
    # transforma em RMSE:
    resultados_df['RMSE'] = resultados_df['MSE']**(1/2)
    resultados_df = resultados_df.sort_values('MSE')

    resultados_df = resultados_df[['RMSE', 'relative error', 'run time']]

    print('ordenado por RMSE:')
    print(resultados_df)
    print('')
    print('ordenado por relative error:')
    print(resultados_df.sort_values('relative error'))

    end = timer()
    print('Tempo total para análise:', timedelta(seconds=end-start))

    # formatacao para a dissertacao:
    resultados_to_print = resultados_df[['RMSE']].T
    resultados_to_print = resultados_to_print[['OLS', 'adaLASSO', 'SVR', 'RF', 'STR RF', 'BooST', 'adaLASSO + RF', 'adaLASSO + STR RF']] # reordena as colunas.
    print(resultados_to_print.round(4).to_latex())

# grava resultados:
data_path = '/home/alega/dissertacao/results/'
with open(data_path + 'qty_of_tress' + '.pkl', 'wb') as handle:
    if number_of_trees_analysis:
        pickle.dump(resultados, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        pickle.dump(resultados, handle, protocol=pickle.HIGHEST_PROTOCOL)

if number_of_trees_analysis:
    path = '/home/alega/dissertacao/results/' + 'qty_of_trees' + '.csv'
    resultados_df.to_csv(path)
else:
    path = '/home/alega/dissertacao/results/' + dataset + '.csv'
    resultados_df.to_csv(path)

##############
# FIM DAS SIMULACOES
##############