# QRT DataChallenge 2022

This repository contains the code I used for my participation in the QRT 2022 Data Challenge [link](https://challengedata.ens.fr/challenges/72) 

### Table of Contents

- [QRT DataChallenge 2022](#qrt-datachallenge-2022)
    - [Table of Contents](#table-of-contents)
  - [Description challenge (copy paste from official website)](#description-challenge-copy-paste-from-official-website)
    - [1. Challenge context](#1-challenge-context)
    - [2. Challenge Goal](#2-challenge-goal)
    - [3. Output structure.](#3-output-structure)
  - [What I implemented](#what-i-implemented)
    - [1. Idea](#1-idea)
    - [2. Ranking](#2-ranking)
    - [3. Quick start](#3-quick-start)
    - [4. Todo](#4-todo)

## Description challenge (copy paste from official website)

### 1. Challenge context
A classic prediction problem from finance is to predict the next returns (i.e. relative price variations) from a stock market. That is, given a stock market of $\mathbb{N}$ stocks having returns $R_t\in\mathbb{R^N}$  at time $t$ the goal is to design at each time $t$ a vector $S_{t+1}\in\mathbb R^N$
from the information available up to time $t$ such that the prediction overlap $\langle S_{t+1},R_{t+1}\rangle$ is quite often positive. To be fair, this is not an easy task. In this challenge, we attack this problem armed with a linear factor model where one learns the factors over an exotic non-linear parameter space.

More precisely, the simplest estimators being the linear ones, a typical move is to consider a parametric model of the form

$S_{t+1}:=\sum_{\ell=1}^F \beta_\ell \, F_{t,\ell}$ where the vectors $F_{t,\ell}\in\mathbb R^NF$

are explicative factors (a.k.a. features), usually designed from financial expertise, and $\beta_1,\ldots,\beta_F\in\mathbb R$ are model parameters that can be fitted on a training data set.

**But how to design the factors $F_{t,\ell}$ ?**

if you know no finance and have developed enough taste for mathematical elegance, you may aim at learning the factors themselves within the simplest class of factors, namely linear functions of the past returns:

$F_{t,\ell}:=\sum_{k=1}^{D} A_{k\ell} \, R_{t+1-k}$

 for some vectors $A_\ell:=(A_{k\ell})\in\mathbb R^D$ and a fixed time depth parameter $D$
Well, we need to add a condition to create enough independence between the factors, since otherwise they may be redundant. One way to do this is to assume the vectors $A_\ell$
's are orthonormal, $\langle A_k,A_\ell\rangle = \delta_{kl}$ 
​for all $k$,$\ell$ , which adds a non-linear constraint to the parameter space of our predictive model.

All in all, we thus have at hand a predictive parametric model with parameters:

* a $D\times F$ matrix $A:=[A_1,\ldots,A_F]$ with orthonormal columns,
* a vector $\beta:=(\beta_1,\ldots,\beta_F)\in R^F$

### 2. Challenge Goal

The goal of this challenge is to design/learn factors for stock return prediction using the exotic parameter space introduced in the context section.

Participants will be able to use three-year data history of $50$ stock from the same stock market (training data set) to provide the model parameters $(A,\beta)$ as outputs. Then the predictive model associated with these parameters will be tested to predict the returns of $50$ other stocks over the same three-year time period (testing data set).

We allow $D=250$ days for the time depth and $F=10$ for the number of factors.

**Metric**. More precisely, we assess the quality of the predictive model with parameters $(A,\beta)$ as follows. Let $\tilde R_t\in R^{50}$ be the returns of the 50 stocks of the testing data set over the three-year period $(t=0\ldots753)$ and let $\tilde S_{t} = \tilde S_{t}(A,\beta)$ be the participants' predictor for $\tilde R_{t} $. The metric to maximize is defined by

$\mathrm{Metric}(A,\beta):= \frac 1{504}\sum_{t=250}^{753} \frac{\langle \tilde S_{t}, \tilde R_{t}\rangle}{\|\tilde S_{t}\|\|\tilde R_{t}\|}$

if $|\langle A_i,A_j\rangle-\delta_{ij}|\leq 10^{-6}$ 
  for all $i,j$ and $\mathrm{Metric}(A,\beta):=-1$ otherwise.

By construction the metric takes its values in $[-1,1]$ and equals to -1−1 as soon as there exists a couple (i,j)(i,j) breaking too much the orthonormality condition.

### 3. Output structure. 

The **output** expected from the participants is a vector where the model parameters $A=[A_1,\ldots,A_{10}]\in\mathbb R^{250\times 10}$ 
  and $beta\in R^{10}$ 

$\text{Output} = \left[\begin{matrix} A_1 \\ \vdots \\ A_{10} \\ \beta \end{matrix}\right]\in\mathbb R^{25 \times 10}$

## What I implemented

### 1. Idea


My best bet is to use `torch` as an optimization tool.
I used `torch` not to build a deep learning solution because the challenge structure forces the model to be linear.

Therefore, the central idea of ​​my solution is to optimize the model with all the constraints given in the data challenge.

In a nutchell I tried to optimize the output parameters given the constraints that we had using pytorch built-in functionalities:
* a. I enforced  A matrix orthogonality using product of Householder reflectors
* b. During optimization I defined "one sample" as all the information from one time period with all the returns for all the stocks
* c. I used torch CosineEmbeddingLoss as the loss to be optimized
* d. Combining a, b, c allows to reproduce a metric close to the one that we want maximize for the challenge
* e. Finaly I used pytorch lightening and ray tune for hyperparameters optimization during the cross validation

### 2. Ranking

The solution was ranked 2nd in the public leaderboard but only 78th on the private leaderboard.
There is a lot of inconsistency in the two leaderboards (1st public LB -> 81st private LB, 3rd public LB -> 74th private LB/ 99th public LB -> 1st private LB, 29th public LB -> 2nd private LB ...) that can certainly be explained by the difference between public/private test sets and by the fact that a lot of provided solutions were quite close (as we are submitting the parameters of a model).

### 3. Quick start

Installation is fairly simple, the code is running on python 3.9, required modules can be installed using 

```console
pip install -r requirements.txt
```

The code can be used by running main.ipynb (note: `data_dir` should be overwritten with the path where data are located)

### 4. Todo 

In order to improve the results I think some ideas could have been explored:

* Knowledge distillation (=building a more complex teacher model, then train the model with the constrainted structure to mimick this complex model)
* Train with negative samples in the cosine embedding loss
* Ensembling could be used in this contexte (fixing $A$ matrix then we could use bagging or other ensembling methods to improve $\beta$)
* Outliers could have been removed before training the pipeline