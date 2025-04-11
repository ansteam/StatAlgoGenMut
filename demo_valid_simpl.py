#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 15:04:36 2021

@author: serlet
"""

#*******************************************************************************
# COPYRIGHT : F. Kwiatkowski, L. Serlet, A. Stos, Clermont Auvergne university *
#*******************************************************************************


#**********************************************************
#**           COMMERCIAL USE IS PROHIBITED               **
#** THIS SOFTWARE IS FOR SCIENTIFIC RESEARCH ONLY        **
#**********************************************************


#**********************************************************
#**           DISTANCE MINIMIZATION ALGORITHM            **
#**                TEST ON SIMULATED DATA                **
#**                  SIMPLIFIED VERSION                  **
#**********************************************************

# This software is a demonstration program to test the validity of the distance minimization algorithm
# for parametric estimation and model selection, on simulated data

# This software performs different tasks. See companion article for more explanations. 
# First, producing dataset as set of families (genotype + phenotype) with prescribed genealogical structure.
# These data can be produced according to three models : no mutation ,single mutation, double mutation
# Secondly, for each (simulated) dataset find to distance to each of the three models 
# that is the closest distance --according to certain statistics-- between the simulated dataset
# and replicas generated with certain values of the parameters of the model (see below)
# Finally choose the model which achieve smallest distance.


# Genotypes are generated but hidden during analysis. Phenotypes are the visible data.
# The phenotype of an individual is the age of onset of the disease K or 200 which mean by convention that K has not occured
# In model 1, the genotype is 1 (mutated) or 0 (wild type) 
# In model 2, the genotype is [0,0] i.e. wild type or [1,0] i.e. having mutation 0
# or [0,1] i.e. having mutation 1  or [1,1] i.e. doubly mutated
# In model 0, no mutation is considered


# Used packages
from time import time
import numpy as np
from numpy.random import rand
import numba as nb

#****************************************
# Specifying the parameters in each model
#****************************************
# The distribution of the age of onset can follow two fixed distribution
# one for mutated (in model 1) or doubly mutated (in model 2) individuals
# the other one for the other cases: wild type or only one mutation in model 2
# In model 0 the distribution of the age of onset is a mixture of these two laws 
# These two distributions are respectively defined below

decla_mut =[0]*10+[0.01]*10+[0.02,0.04,0.08,0.15,0.3]+[0.5,0.7,0.9,1.2]+[1.4,1.8,2,2.3,2.5]+[2.7, 2.9, 3.1, 3.3, 3.5]
decla_mut += [3.6, 3.85, 3.9, 3.85, 3.6]+[3.5,3.4,3.3,3.1,3,2.9,2.8,2.7,2.6,2.5]+[2.4,2.2,2,1.8,1.6]
decla_mut += [1.4,1.2,1, 0.8, 0.75] +[0.7, 0.65, 0.6, 0.55,0.5]+[0.45,0.4,0.35,0.3,0.28]+[0.27,0.25,0.23,0.20,0.17]
decla_mut += [0.15,0.13,0.11,0.09,0.07]+ [0.06, 0.05, 0.04,0.03,0.025]+[0.01]*10+[0.005]
age_declar_mut = np.array([decla_mut[k]/100 for k in range(100)] , dtype=np.float64)

decla_norm = [0]*10+[0.01]*10+ [0.02]*5+ [0.04]*5+ [0.12]*5 +[0.2,0.4,0.5,0.6,0.7]+[0.8,0.9,1,1.1,1.2]
decla_norm += [1.2]*5 + [1.3]*5 +[1.3,1.3,1.4,1.5,1.6]+[1.7,1.8,1.9,2,2.1]+ [2.1,2.22,2.22,2.1,2.1]
decla_norm +=[2.1,2,2,2,1.9] + [1.75]*5+ [1.7]*10 +[1.6]*9+[1.61]
age_declar_norm = np.array([decla_norm[x]/100 for x in range(100)], dtype=np.float64)

# MODEL 0 has 2 parameters gathered in an array as "param":
#        - mixture parameter of the distributions of age of onset, "param[0]"
#        - penetrance for all individuals, "param[1]"
# The domain of variation of these parameters in simulated data is as follows 

ampl_para0 = np.array([[0.01,0.2],[0.01,0.15]], dtype=np.float64)

# MODEL 1 has 3 parameters gathered in an array as "param":
#         - probability of mutation, "param[0]"
#         - penetrance for wild type, "param[1]"
#         - penetrance for mutated type, "param[2]"
# The domain of variation of these parameters in simulated data is as follows

ampl_para1 = np.array([[0.02,0.2],[0.01,0.10],[0.25,0.85]], dtype=np.float64)

# MODEL 2 has 4 parameters gathered in an array as "param":
#         - probability of mutation 0, "param[0]"
#         - probability of mutation 1, "param[1]"
#         - penetrance for non doubly mutated, "param[2]"
#         - penetrance for doubly mutated, "param[3]"

ampl_para2 = np.array([[0.02,0.08],[0.12,0.25],[0.01,0.10],[0.35,0.85]], dtype=np.float64)

# The genealogical tree of a family with N members is a 4 x N array "arbr" 
# line 0 consists of zeros except the first coefficient arbr[0,0]  which is set to N
# line 1 (arbr[1,:]) gives the indexes of the fathers with convention -1 if not in the tree
# line 2 (arbr[2,:]) gives the indexes of the mothers with convention -1 if not in the tree
# line 3 gives the generation rank starting at 0 for the younger individuals

# We use regular trees built by the following function.
# The number of generations is "numb_gene"
# and the number of kids per couple is "numb_kid"
# This tree contains all the ascendants of the individual numbered 0
# Then more individuals are added so that each couple has "numb_kid" children

@nb.jit(nb.int64[:,:](nb.int64, nb.int64), nopython=True)
def arbre_reg(numb_gene, numb_kid):
    # Initializing the lists with the individual 0
    fathe = [1] 
    mothe = [2]
    gener = [0]
    # Creating the individuals along the ascending lines of individual 0
    # except the oldest generation
    for gen in range(1, numb_gene - 1):
        for indig in range(2** gen -1 , 2** (gen +1) -1):
            fathe.append(2*indig +1)
            mothe.append(2*indig +2)
            gener.append(gen)
    # Adding the oldest generation for which the parents are out of the tree
    for indig in range(2**(numb_gene -1)-1 , 2**numb_gene  -1):
        fathe.append(-1)
        mothe.append(-1)
        gener.append(numb_gene -1)
    # Then the skeleton formed by the ascending lines is decorated
    # with added individuals o that each couple has "numb_kid" children
    for gen in range(numb_gene-1 , 0,-1):
        for male in range(2**gen -1, 2**(gen+1)-1, 2):
            fathe = fathe + [male]*(numb_kid-1)
        for female in range(2**gen, 2**(gen+1)-1, 2):
            mothe = mothe + [female]*(numb_kid-1)
        gener = gener + [gen-1]*((2**(gen-1))*(numb_kid-1))
    #Finally convert into the desired array format
    siz = len(fathe)
    lign_siz = [siz] + [0]*(siz-1)
    arbr = np.array([lign_siz, fathe, mothe, gener], dtype = np.int64)
    return arbr


# ********************************************************************
# Generation of the genotype of a family in model 1 
# with a genealogical tree given by "arbr"
# with parameters given by "param"
@nb.jit(nb.int64[:,:](nb.int64[:,:], nb.float64[:]), nopython=True)
def draw_geno1(arbr , param):
    # rename the variables more explicitly
    siz = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    prob_mut = param[0]
    # vector matrix for the probability of being wild type
    # choose line according to the sum of the parents genotypes
    mendel_2par = np.array( [1, 0.5, 0.25 ], dtype = np.float64)
    # vector matrix for the probability of being wild type
    # when only one parent is known 
    # choose line 0 or 1 according to the parent's genotype
    mendel_1par = (1 - prob_mut) * np.array([1, 0.5])
    mendel_1par += prob_mut * np.array([0.5, 0.25])
    # initializing the genotype variable 
    # at the end it will contain the result in its  column 0
    # useless  column 1 is for standardization with model 2
    work_geno1 = 2 * np.ones((siz,2), dtype=np.int64)
    # initializing the number of treated individuals in the tree
    numb_done1 = 0
    # variable scanning the tree, possibly several times
    scan1 = 0
    # let it run until all individuals have been treated
    while numb_done1 < siz:
        # test if the current individual, indexed by scan1, has undetermined genotype
        # otherwise do nothing
        if work_geno1[scan1,0] == 2 :
            # 1st case: parents not in the tree
            if moth[scan1] == -1 and fath[scan1] == -1 :
                if rand() < 1 - prob_mut:
                    work_geno1[scan1,0] = 0
                else: 
                    work_geno1[scan1,0] = 1
                numb_done1 += 1
            # 2nd case: mother out of the tree, father in the tree with determined genotype
            if moth[scan1] == -1 and fath[scan1] != -1 and work_geno1[fath[scan1],0] != 2:

                if rand() < mendel_1par[ work_geno1[fath[scan1],0] ]:
                    work_geno1[scan1,0] = 0
                else:
                    work_geno1[scan1,0] = 1
                numb_done1 += 1
            # 3rd case: mother in the tree with determined genotype, father out of the tree
            if moth[scan1] != -1 and fath[scan1] == -1 and work_geno1[moth[scan1],0] != 2:
                if rand() < mendel_1par[ work_geno1[moth[scan1],0] ]:
                    work_geno1[scan1,0] = 0
                else:
                    work_geno1[scan1,0] = 1
                numb_done1 += 1
            # 4th case: both parents in the tree with determined genotypes
            if moth[scan1] != -1 and fath[scan1] != -1 \
                   and work_geno1[fath[scan1],0] != 2 and work_geno1[moth[scan1],0] != 2 :
                # computing the number of mutated parents
                sumpar =  work_geno1[fath[scan1],0] +  work_geno1[moth[scan1],0]
                if rand() < mendel_2par[sumpar]:
                    work_geno1[scan1,0] = 0
                else:
                    work_geno1[scan1,0] = 1
                numb_done1 += 1
            # passing to the next individual
        scan1 = (scan1 + 1) % siz
    return work_geno1



#********************************************************************
# Generation of the genotype of a family in model 2
# with a genealogical tree given by "arbr"
# with parameters given by "param"
# the result is an array with as many lines as the size of "arbr"
# and two columns corresponding resp. to mutation 0 and 1
@nb.jit(nb.int64[:,:](nb.int64[:,:], nb.float64[:]), nopython=True)
def draw_geno2(arbr, param) :
    # rename the variables more explicitely 
    siz = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    prob_2mut_0 = param[0]
    prob_2mut_1 = param[1]
    # vector matrix for the probability of being wild type
    # choose line according to the sum of the parents genotypes
    mendel_2par = np.array( [1, 0.5, 0.25], dtype = np.float64)
    # vector matrix for the probability of being wild type
    # when only one parent is known 
    # as in  function "draw_geno1"
    # one for each of the two possible mutations
    mendel_1par_mut0 = (1 - prob_2mut_0) * np.array( [1, 0.5], dtype = np.float64)
    mendel_1par_mut0 += prob_2mut_0 * np.array( [0.5, 0.25], dtype = np.float64)
    mendel_1par_mut1 = (1 - prob_2mut_1) * np.array( [1, 0.5], dtype = np.float64)
    mendel_1par_mut1 += prob_2mut_1 * np.array( [0.5, 0.25], dtype = np.float64)
    # initializing the genotype variable
    # at the end column 0 will  contain status for mutation 0
    # and column 1 will  contain status for mutation 1
    work_geno2 = 2 * np.ones((siz, 2), dtype=np.int64)
    # initializing the number of treated individuals in the tree
    numb_done2 = 0
    # variable scanning the tree, possibly several times
    scan2 = 0
    while numb_done2 < siz:
        # test if the current individual, indexed by scan2, has undetermined genotype 
        if np.sum(work_geno2[scan2, :]) == 4:
            # same successive cases as for model 1
            # 1st case: parents not in the tree
            if moth[scan2] == -1 and fath[scan2] == -1 :
                if rand() <  prob_2mut_0:
                    gen0 = 1
                else:
                    gen0 = 0
                if rand() <  prob_2mut_1:
                    gen1 = 1
                else:
                    gen1 = 0
                work_geno2[scan2, :] = np.array([gen0, gen1])
                numb_done2 += 1
            # 2nd case: mother out of the tree, father in the tree with determined genotype
            if moth[scan2] == -1 and fath[scan2] != -1 and np.sum(work_geno2[fath[scan2], :]) < 4:
                # Decide on mutation 0, taking the status of the father into account   
                if rand() < mendel_1par_mut0[work_geno2[fath[scan2], 0] ] :
                    gen0 = 0
                else:
                    gen0 = 1
                # Then identically for mutation 1
                if rand() < mendel_1par_mut1[work_geno2[fath[scan2], 1] ] :
                    gen1 = 0
                else:
                    gen1 = 1
                # Compiling
                work_geno2[scan2, :] = np.array([gen0, gen1])
                numb_done2 += 1
            # 3rd case: mother in the tree with determined genotype, father out of the tree
            if moth[scan2] != -1 and fath[scan2] == -1 and np.sum(work_geno2[moth[scan2], :]) < 4:
                if rand() < mendel_1par_mut0[work_geno2[moth[scan2], 0] ]:
                    gen0 = 0
                else:
                    gen0 = 1
                if rand() < mendel_1par_mut1[work_geno2[moth[scan2], 1] ]:
                    gen1 = 0
                else:
                    gen1 = 1
                work_geno2[scan2, :] = np.array([gen0, gen1])
                numb_done2 += 1
            #4th case: both parents in the tree and determined
            if moth[scan2] != -1 and fath[scan2] != -1 \
                and np.sum(work_geno2[fath[scan2],:]) < 4 and np.sum(work_geno2[moth[scan2],:]) < 4:
                # computing the number pf parents holding mutation 0
                # could be 0 or 1 or 2
                sum0 =  work_geno2[fath[scan2], 0] +  work_geno2[moth[scan2], 0]
                if rand() < mendel_2par[sum0] :
                    gen0 = 0
                else:
                    gen0 = 1
                # Then identically for mutation 1
                sum1 =  work_geno2[fath[scan2], 1] +  work_geno2[moth[scan2], 1]              
                if rand() < mendel_2par[sum1] :
                    gen1 = 0
                else:
                    gen1 = 1
                work_geno2[scan2, :] = np.array([gen0, gen1])
                numb_done2 += 1
        # Passing to the next individual by incrementing the index
        scan2 = (scan2 + 1) % siz
    return work_geno2


#*****************************************************************
# Generation of the phenotype for a family in any of the 3 models
# as specified by model = 0 or model = 1 or model = 2
# with a genealogical tree given by "arbr"
# with parameters given by "param"
# and with a  genotype specified by "geno"
# which is an array with two columns.
# The law of the age of onset is given by "adec_mut"
# in the mutated  (model =1) or doubly mutated case (model=2)
# and by "adec_norm" in the other cases.
# The result is an array with 2 columns;
# column 0 is the age of onset if the disease K has occured and 200 otherwise;
# column 1 is for the age or potential age, generated according to generations.
@nb.jit(nb.int64[:,:](nb.int64, nb.int64[:,:], nb.int64[:,:], nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
def phenotype(model, arbr, geno, param, adec_norm, adec_mut):
    siz = arbr[0,0]
    gener = arbr[3,:]
    # builing the distribution of the age of onset for use in each model
    adec0 = param[0] * adec_norm + (1 - param[0]) * adec_mut
    adec1 = np.vstack((adec_norm, adec_mut))
    adec2 = np.vstack((adec_norm, adec_norm, adec_mut))
    #for model = 2, creating a 2x2 matrix giving the penetrance 
    # where status of mutation 0 is the index 0 (line index)
    # and status of mutation 1 the index 1 (column index)
    if model == 2 :
        pene2 = np.array( [param[2], param[2], param[2], param[3]] ).reshape(2,2)
    # initializing the desired phenotype
    phenoty = np.zeros( (siz, 2), dtype = np.int64)
    # Scanning the tree
    for i in range(siz):
        # Set the (potential) age according to the generation
        if gener[i] == 0:
            age_pot = np.random.randint(30, 51)
        if gener[i] == 1:
            age_pot = np.random.randint(50, 71)
        if gener[i] == 2:
            age_pot = np.random.randint(70, 91)
        if gener[i] >= 3:
            age_pot = 99
        # Determine the probability that the disease K has occured
        # and the law of the age of onset, up to the age of the individual
        if model == 0:
            normali = np.sum(adec0[ 0 : age_pot])
            proba_K = min (param[1] * normali, 0.9999)
            vect_proba = adec0[ 0 : age_pot] / normali
        if model == 1:
            normali = np.sum( adec1[ geno[i,0], 0 : age_pot] )
            proba_K = min( param[ geno[i,0] + 1]  * normali, 0.9999)
            vect_proba = adec1[ geno[i,0], 0:age_pot] / normali
        if model == 2:
            sum_geno = np.sum(geno[i, :])
            normali = np.sum(adec2[sum_geno, 0:age_pot])
            vect_proba = adec2[ sum_geno, 0:age_pot ] / normali 
            proba_K = min( pene2[ geno[i,0], geno[i,1] ] * normali, 0.9999)
        if rand() > proba_K :
            # no K, the age of onset is 200
            phenoty[i,0] = 200
            phenoty[i,1] = age_pot
        else:
            # age of onset has to be generated according to the vector "vect_proba"
            ff = np.cumsum(vect_proba)
            ff[-1] = 1
            j = 0
            ch = rand()
            while ch > ff[j]:
                j += 1
            # the last value of j is the desired random age of onset
            # in this cas the age is set to 200, by convention
            phenoty[i,0] = j
            phenoty[i,1] = 200
    return phenoty


#*******************************************************************
# Notion of proximity between to phenotypes
# being both  1x2 arrays 
# The possible results are -3, 1, 5, 10
# meaning increasing proximities
@nb.jit(nb.int64( nb.int64[:], nb.int64[:]), nopython = True)
def proxi(phen1, phen2):
    # sco = 0
    if phen1[0] == 200 and phen2[0] == 200 :
        # no K for both, common situation
        return 1
    if phen1[0] < 200 and phen2[0] == 200 :
        # K for one only
        return -3
    if phen1[0] == 200 and phen2[0] < 200 :
        return -3
    if phen1[0] < 200 and phen2[0] < 200 :
        # K for both
        if phen1[0] < 50 and phen2[0] < 50:
            # both at early ages
            return 10
        else:
            if abs(phen1[0] - phen2[0]) < 11:
                # K at similar ages
                return 5
            else:
                return 3
    return 0

#****************************************************************
# Computing one of the statistics under consideration,
# a proximity score between the individuals and their parents and grandparents
@nb.jit(nb.float64( nb.int64[:,:], nb.int64[:,:] ), nopython = True)
def score_par(arbr, pheno): 
    siz = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    # Initializing the desired score
    scor_par = 0
    # Initializing the total number of parents
    # and grandparents that will be considered
    numb_par_gdpar = 0
    # scanning individuals in the tree
    for ind in range(siz):
        # Initializing two arrays that will list
        # the parents and grandparents respectively
        parents = -1 * np.ones(siz, dtype = np.int64)
        gdparents = -1 * np.ones(siz, dtype = np.int64)
        # Initializing two variables that will count
        # the parents and grandparents respectively
        par_count = 0
        gdpar_count = 0
        # If the scanned individual has a father
        if fath[ind] != -1:
            # he has to be added to the parent's array
            parents[par_count] = fath[ind]
            numb_par_gdpar += 1
            par_count += 1
            # if this father has a father
            if fath[fath[ind]] != -1:
                # we make an addition to the list of grandparents
                gdparents[gdpar_count] = fath[fath[ind]]
                numb_par_gdpar += 1
                gdpar_count += 1
            if moth[fath[ind]] != -1:
                # similarly for a mother
                gdparents[gdpar_count] = moth[fath[ind]]
                numb_par_gdpar += 1
                gdpar_count += 1
        # identically for the mother side of individual "ind"
        if moth[ind] != -1:
            parents[par_count] = moth[ind]
            numb_par_gdpar += 1
            par_count += 1
            if fath[moth[ind]] != -1:
                gdparents[gdpar_count] = fath[moth[ind]]
                numb_par_gdpar += 1
                gdpar_count += 1
            if moth[moth[ind]] != -1:
                gdparents[gdpar_count] = moth[moth[ind]]
                numb_par_gdpar += 1
                gdpar_count += 1
        # Computing the cumulative proximity with the parents
        for ipar in range(par_count):
            scor_par +=  proxi(pheno[ind,:], pheno[parents[ipar],:])
        # Computing the cumulative proximity with the grandparents
        for igdpar in range(gdpar_count):
            scor_par +=  proxi(pheno[ind,:], pheno[gdparents[igdpar],:])
    # when all members have been scanned, the score is normalized       
    return scor_par /  numb_par_gdpar

#****************************************************************
# Computing another statistics under consideration,
# a proximity score between the individuals and their sibblings
@nb.jit(nb.float64( nb.int64[:,:], nb.int64[:,:]), nopython=True)
def score_frat(arbr, pheno):
    siz = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    scor_frat = 0
    numb_tot_frat = 0
    for ind in range(siz):
        # We initialize an array that will contain the indexes
        # of the brother(s)/sister(s) of the scanned individual "ind"
        brothers = np.zeros(siz, dtype=np.int64)
        bro_count_ind = 0
        # We restrict to the case where parents are in the tree
        if fath[ind] != -1 and moth[ind] != -1 and ind < siz - 1 :
            # scanning all individuals with higher index
            for pot_bro in range(ind+1, siz):
                # We test if the parents are identical
                if fath[pot_bro] == fath[ind] and moth[pot_bro] == moth[ind]:
                    brothers[bro_count_ind] = pot_bro
                    bro_count_ind += 1
                    numb_tot_frat += 1
        for bro_ind in range(bro_count_ind):
            # We add the proximity score for all siblings
            scor_frat +=  proxi(pheno[ind,:], pheno[brothers[bro_ind],:])
    if numb_tot_frat == 0:
        return 0
    else:
        return scor_frat / numb_tot_frat


# *****************************************************************
# This function computes the statistical summary coressponding to
#   - a specified model (0 or 1 or 2)
#   - a specified tree shape "arbr"
#   - a specified value of the parameters "param" 
#   - a specified minimal number of disease case per family "loc_min_K"
#   - specified distributions of the age of onset "adec_norm" and "adec_mut"
#               corresponding to the mutational status
# It is performed by simulating  "loc_numb_simu" families with the specified genalogical tree
# For the present version, the summary consists in 6 statistics
#   (1)  frequency of K among all the individuals  in the tree
#   (2)  mean age of onset 
#   (3)  among the individuals for who K has occured and whose parents are in the tree, 
#             proportion of case where K has not occured for either of the parents
#   (4)  among the individuals where K has occured and whose parents are in the tree, 
#             mean number of K concerning the parents
#   (5) proximity score with the parents and grandparents
#   (6) proximity score with the siblings
@nb.jit(nb.float64[:](nb.int64, nb.int64[:,:], nb.float64[:], nb.int64, nb.int64, nb.float64[:], nb.float64[:]), nopython=True)
def summ_stat(model, arbr, param, loc_min_K, loc_numb_simu, adec_norm, adec_mut):
    siz = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    # initializing the counting variables
    loc_numb_K = 0.0
    loc_cumul_age_K = 0.0
    loc_numb_K_2par = 0.0
    loc_numb_K_isol = 0.0
    loc_cumul_K_par = 0.0
    loc_scor_par = 0.0
    loc_scor_frat = 0.0
    loc_numb_2par = 0.0
    #  initializing the statistical summary as an array 
    loc_result = np.zeros(6, dtype=np.float64)
    nozero = False
    count_sim = 0
    # the first loop is to ensure the desired number of simulations 
    # and the non nullity of denominators
    while (count_sim < loc_numb_simu or nozero == False):
        number_K = 0
        # we use rejection sampling
        # to ensure that generated phenotypes have at least loc_min_K cases of K
        while number_K < loc_min_K :
            # First generate the genotype in accordance to the model
            if model == 1:
                loc_geno = draw_geno1(arbr, param)
            if model == 2:
                loc_geno = draw_geno2(arbr, param)
            if model == 0:
                loc_geno = np.zeros((siz,2), dtype=np.int64)
                # in this case the genotype is irrelevant
            # Then generate the phenotype
            loc_pheno = phenotype(model, arbr, loc_geno, param, adec_norm, adec_mut)
            # Counting the number of K
            number_K = siz - np.sum(loc_pheno[:, 0] == 200)
        
        # A suitable sample has been drawn, let us compute its contribution to the statistical summary
        loc_numb_K += number_K
        for indi in range(siz):
            # Adding all the ages of onset
            if loc_pheno[indi, 0] < 200:
                loc_cumul_age_K += loc_pheno[indi, 0]
            # Case with both parents in the tree
            if fath[indi] != -1 and moth[indi] != -1 :
                loc_numb_2par += 1
                if loc_pheno[indi, 0] < 200:
                    loc_numb_K_2par += 1
                    if loc_pheno[fath[indi], 0] == 200 and loc_pheno[moth[indi], 0] == 200:
                        loc_numb_K_isol += 1
                    else:
                        if loc_pheno[fath[indi], 0] == 200:
                            loc_cumul_K_par += 1
                        if loc_pheno[moth[indi], 0] == 200:
                            loc_cumul_K_par += 1
        loc_scor_par += score_par(arbr, loc_pheno)
        loc_scor_frat += score_frat(arbr, loc_pheno)
        if loc_numb_K!= 0 and loc_numb_K_2par != 0 :
            if loc_numb_K_isol != 0 and loc_cumul_K_par != 0:
                nozero = True
        count_sim += 1
    # compiling the results
    loc_result[0] = loc_numb_K  / (siz *  count_sim)
    loc_result[1] = loc_cumul_age_K /  loc_numb_K
    loc_result[2] = loc_numb_K_isol / loc_numb_K_2par 
    loc_result[3] = loc_cumul_K_par / loc_numb_K_2par 
    loc_result[4] = loc_scor_par / count_sim
    loc_result[5] = loc_scor_frat / count_sim
    return loc_result

#*************************************************************************
# Computing the distance to each of the models 
# and obtaining th best parameters 
# for a dataset completely summarized by its summary "sta_summ"
# The algoritm below seeks the set of  parameters 
# for which statistical summary is the closest to "sta_summ"
# Parameters of this function have the same meaning as usual
# "ampl_para" defines in what intervals the optimal values of parameters are searched
@nb.jit(nb.float64[:,:]( nb.int64, nb.float64[:], nb.int64[:,:],  nb.int64, nb.float64[:], nb.float64[:], nb.float64[:,:]), nopython=True)
def dist_to_mod(model, sta_summ, arbr, min_K, adec_norm, adec_mut, ampl_para):
    # Introduce the dimension of the statistical summary
    dim_summ = len(sta_summ)
    # Introduce the dimension of the parameters as a function of the model
    if model == 0 :
        dim_para = np.int64(2)
    if model == 1 :
        dim_para = np.int64(3)
    if model == 2 :
        dim_para = np.int64(4)
    #dim_para = np.int64(np.shape(ampl_para)[0])
    # Total varying dimension
    dim = dim_para + dim_summ
    # We let the possibility to introduce coefficients in the distance
    coeff = np.ones(  dim_summ, dtype = np.float64)
    # First search the minimum on a grid
    # Choose the number of points to form this grid
    nbr_maill_para =  np.ones( dim_para, dtype = np.int64) * 10
    # Setting the number of simulation in the first and second phases
    numb_simu_ini = np.int64(100)
    numb_simu_refin = np.int64(400)
    # Setting the number of kept values
    size_best = 8
    # We will gather the best parameter values in an array "tabl_first_best"
    # "size_best" is the number of selected values 
    # and the number of lines in the array
    # The array has dim + 1 columns, from 0 to dim:
    #     - columns 0 to dim_data excluded  are for the parameters values
    #     - columns dim_data to dim excluded for the corresponding summaries
    #     - column dim for the distance
    # Choosing how many of the best values will be collected 

    tabl_first_best = np.ones(( size_best + 1, dim + 1), dtype = np.float64) * 100
    temporary  = np.zeros( dim + 1, dtype = np.float64 )
    # To browse the grid, initiate an array 
    curr = np.zeros ( dim_para,  dtype = np.int64)
    new = np.zeros ( dim_para,  dtype = np.int64)
    curr_para = np.zeros ( dim_para,  dtype = np.float64)
    while  curr[0]  < nbr_maill_para[0] :
        # from the "curr" array, create an array of the  parameter values under consideration
        for i in range(dim_para):
            curr_para[i] = ampl_para[i,0]
            curr_para[i] += ( ampl_para[i,1] - ampl_para[i,0] ) * curr[i] / nbr_maill_para[i]
        curr_stat =   summ_stat(model , arbr, curr_para, min_K, numb_simu_ini, adec_norm, adec_mut)
        # Distance computation
        curr_dist = np.float64(0)
        for i in range(dim_summ):
            if curr_stat[i] != 0:
                curr_dist += coeff[i] * ( max(curr_stat[i] / sta_summ[i], sta_summ[i] / curr_stat[i] ) - 1 ) ** 2
            else:
                # this case is normally excluded.
                curr_dist += 2
        curr_dist =  np.sqrt(curr_dist)
        # Keeping in the array "tabl_first_best" 
        # the "size_best" best value by increasing order of distance
        if curr_dist < tabl_first_best[size_best - 1, dim]:
            tabl_first_best[size_best, 0 : dim_para] = curr_para[:]
            tabl_first_best[size_best, dim_para : dim] = curr_stat
            tabl_first_best[size_best, dim] = curr_dist
            smaller = True
            rank = size_best - 1
            while smaller == True:
                temporary[:]= tabl_first_best[rank,:]
                tabl_first_best[rank, : ] = tabl_first_best[rank + 1, : ]
                if rank < size_best - 1:
                    tabl_first_best[rank + 1, : ] = temporary[ : ]
                if rank == 0:
                    smaller = False
                else:
                    if tabl_first_best[rank - 1, dim] <  tabl_first_best[rank, dim]:
                        smaller = False
                    else:
                        rank -= 1
            #print(tabl_first_best[:, dim])
        new [:] = curr[:]
        new[ dim_para - 1] += 1
        for j in range(dim_para - 1, 0, -1) :
            if new[j] == nbr_maill_para[j]:
                new[j-1] += 1
                new[j] = 0
        curr[:] = new[:]
    # We then begin the second phase of refined search in the neighborhood of 
    # each parameter values previously selected with more precise simulation
    tabl_final_best = np.ones( ( dim + 1), dtype = np.float64 ) * 10
    # Number of tries around the selected value
    loc_try_refin = 10
    refin_para = np.zeros(dim_para, dtype = np.float64)
    for scan_first in range(size_best) :
        for loc_refin in range(loc_try_refin) :
            if loc_refin == 0:
                refin_para = tabl_first_best[scan_first, 0 : dim_para]
            else:
                # try little variation sof the parameters
                refin_alea = np.random.rand(dim_para)
                pas = np.float64(0)
                new_para = np.float64(0)
                for j in range(dim_para):
                    pas = (ampl_para[j,1] - ampl_para[j,0]) / ( nbr_maill_para[j] * (1+np.sqrt(np.float64(loc_refin)))  )
                    new_para = tabl_first_best[scan_first,j]+ pas * (2 * refin_alea[j] - 1)
                    refin_para[j] = min(max(new_para, ampl_para[j,0]), ampl_para[j,1])
            refin_stat =  summ_stat(model, arbr, refin_para, min_K, numb_simu_refin, adec_norm, adec_mut)
            #Distance evaluation
            refin_dist = np.float64(0)
            for i in range(dim_summ):
                if refin_stat[i] != 0:
                    refin_dist += coeff[i] * ( max(refin_stat[i]/sta_summ[i], sta_summ[i]/refin_stat[i] ) - 1 ) ** 2
                else:
                    refin_dist += 2
            refin_dist =  np.sqrt( refin_dist )
            # Keeping a new table "tabl_second_best" of the best values found
            # rank in decreasing order of distance
            if refin_dist < tabl_final_best[ dim]:
                tabl_final_best[ 0 : dim_para] = refin_para[:]
                tabl_final_best[ dim_para : dim] = refin_stat
                tabl_final_best[ dim] = refin_dist
                
            
    
    # The result of the function is a 3 x dim_summ array
    result_dist = np.zeros( (3, dim_summ), dtype = np.float64)
    # On line 0, only the first coefficient is non null
    # and it is the distance
    result_dist[0, 0] = tabl_final_best[dim]  
    # On line 1, the best parameters ( first columns)
    result_dist[1, 0 : dim_para] = tabl_final_best[ 0 : dim_para]
    # On line 2, the corresponding best statistical summary
    result_dist[2, 0 : dim_summ] = tabl_final_best[dim_para : dim]
    return result_dist
    
       





# ********************************************************************************************
# ********************************************************************************************
# Global parameters 
np.random.seed(33)

# Minimal number of disease case in each simulated family
min_K = 1
# Number of families in the initial (simulated) dataset
numb_data = 100
# Number of datasets simulated
numb_essai = 10

# ********************************************************************************************
# ********************************************************************************************



print()
print("**************************************************")
print("*  TESTING THE DISTANCE MINIMIZATION ALGORITHM   *")
print("*   Model selection and parametric estimation    *")
print("*                                                *")
print("* Copyright : F. Kwiatkowski, L. Serlet, A. Stos *")
print("**************************************************")
print()
print()



print("Version based on a statistical summary containing only 6 statistics:")
print()
print("     (1)  frequency of K")
print("     (2)  mean age of onset")
print("     (3)  among the individuals for whom K has occured and whose parents are in the tree,")
print("                proportion of case where K has not occured for either of the parents")
print("     (4)  among the individuals for whom K has occured and whose parents are in the tree,")
print("                mean number of K concerning the parents")
print("     (5)  proximity score with the parents and grandparents")
print("     (6)  proximity score with the siblings")
print()
print("Three models are under scrutiny:")
print("     Model 0 : no  mutation and 2 parameters")
print("     Model 1 : one mutation and 3 parameters")
print("     Model 2 : two mutation and 4 parameters")
print()
print("Minimal number of K per family : ", min_K)
print("Number of families in the dataset :  ", numb_data)



# Different family size can be tried 
for numb_gener in [3]:
    for numb_kids in [2]:
        print()
        print("******************************")
        print("Number of generations : ", numb_gener )
        print("Number of children per couple : ", numb_kids )
        print()
        # Tree generation
        arbre = arbre_reg(numb_gener, numb_kids)
        #****************************
        #Initializing the confusion table
        confus = np.zeros((3,3), dtype=np.float64)
        
        #*******************************************************************
        # GENERATION OF DATASET WITH THE ONE MUTATION MODEL
        # initializing the performance variable
        cumul_dist_para_1mut = 0
        cumul_dist_stat_1mut = 0
        #*****************************
        for essai in range(numb_essai):
            # Drawing the parameters of the dataset  at random
            alea_data = np.random.rand(3)
            para_data = np.zeros(3, dtype=np.float64)
            for j in range(3):
                para_data[j] = ampl_para1[j,0] + alea_data[j] * (ampl_para1[j,1]- ampl_para1[j,0])
            # Compute the statistical summary of the dataset
            stat_data = summ_stat(1, arbre, para_data, min_K, numb_data, age_declar_norm, age_declar_mut)
            # For the first dataset printing the progress of the algorithm
            if essai  == 0:
                print()
                print('**********************************************************************************')
                print("Generating a dataset in the one mutation model")
                print("   with parameters: ", np.round(para_data, 4))
                print("   giving the statistical summary: ", np.round(stat_data, 4) )
                print()
            
            # Estimating the distance to model 0
            time_start = time()
            dist_0mut = dist_to_mod(0, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para0)
            time_end = time()
            if essai   == 0 :
                print()
                print('Distance to model 0: ', np.round(dist_0mut[0,0], 4) )
                print('    attained with the two parameters:    ', np.around(dist_0mut[1,:2], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_0mut[2,:], 4) )
                print('    to be compared with real summary:    ',  np.around(stat_data, 4) )
                print("    Computing time:  ", time_end-time_start)
                print()
            
            # Estimating the distance to model 1
            time_start = time()
            dist_1mut = dist_to_mod(1, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para1)
            time_end = time()
            
            if essai   == 0 :
                print()
                print('Distance to model 1: ', np.round(dist_1mut[0,0], 4) )
                print('    attained with the three parameters:  ', np.around(dist_1mut[1,:3], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_1mut[2,:], 4) )
                print('    to be compared with real summary:    ',  np.around(stat_data, 4) )
                print("    Computing time: ", time_end-time_start)
                print()
        
            # Estimating the distance to model 2
            time_start = time()
            dist_2mut = dist_to_mod( 2, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para2)
            time_end = time()
            if essai   == 0 :
                print()
                print('Distance to model 2: ', round(dist_2mut[0,0], 4) )
                print('    attained with the 4 parameters:      ', np.around(dist_2mut[1,:4], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_2mut[2,:6], 4) )
                print('    to be compared with real summary:    ',  np.around(stat_data, 4) )
                print("    Computing time:  ", time_end-time_start)
                print()
            
            if dist_1mut[0,0] < dist_2mut[0,0] and dist_1mut[0,0] < dist_0mut[0,0]:
                confus[1,1]+= 1
                cumul_dist_stat_1mut += np.sum(np.absolute(1-(dist_1mut[2,:] / stat_data)))
                cumul_dist_para_1mut += np.sum(np.absolute(1-(dist_1mut[1,:3]/para_data)))
            if dist_2mut[0,0] < dist_1mut[0,0] and dist_2mut[0,0] < dist_0mut[0,0]:
                confus[1,2]+= 1
            if dist_0mut[0,0] < dist_1mut[0,0] and dist_0mut[0,0] < dist_2mut[0,0]:
                confus[1,0]+= 1
            if essai != 0 and essai % 10  == 0:
                print('***************************************************')
                print('Number of one mutation dataset treated: ', essai+1)
                print('Accuracy so far: ', confus[1,1] / (essai+1))
                print()
        print()
        print()
        print('********************************************************')
        print('Conclusion : accuracy for one mutation data  = ', np.round(confus[1,1] * 100/numb_essai),' % ')
        if confus[1,1] != 0:
            print('          Mean relative error for parameter estimation: ', 
                      np.round(cumul_dist_para_1mut * 100/ (3*confus[1,1]), 1),' % ' )
            print('          Mean relative error on statistical summary:   ', 
                np.round(cumul_dist_stat_1mut * 100 / (8 * confus[1,1]), 1),' % ')
        print('********************************************************')
    
        
 
        #************************************************************
        # GENERATION OF DATASET WITH THE TWO MUTATIONS MODEL
        cumul_dist_para_2mut = 0
        cumul_dist_stat_2mut = 0
        for essai in range(numb_essai):
            #Tirage au hasard des parametres
            alea_data = np.random.rand(4)
            para_data = np.zeros(4, dtype=np.float64)
            for j in range(4):
                para_data[j] = ampl_para2[j,0]+ alea_data[j] *(ampl_para2[j,1]-ampl_para2[j,0])
            # fabrication des stats des données
            stat_data = summ_stat(2, arbre, para_data , min_K, numb_data, age_declar_norm, age_declar_mut)
            if essai  == 0 :
                
                print()
                print('**********************************************************************************')
                print("Generating a dataset in the two mutations model")
                print("   with parameters: ", np.round(para_data, 4))
                print("   giving the statistical summary: ", np.round(stat_data, 4) )
                print()
                
            
            # Estimating the distance to model 0
            time_start = time()
            dist_0mut = dist_to_mod(0, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para0)
            time_end = time()
            if essai   == 0 :
                
                print()
                print('Distance to model 0: ', np.round(dist_0mut[0,0], 4) )
                print('    attained with the two parameters:    ', np.around(dist_0mut[1,:2], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_0mut[2,:], 4) )
                print('    to be compared with real summary:    ', np.around(stat_data, 4) )
                print("    Computing time: ", time_end-time_start)
                print()
            # Estimating the distance to model 1
            time_start = time()
            dist_1mut = dist_to_mod( 1, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para1)
            time_end = time()
            if essai  == 0 :
                print()
                print('Distance to model 1: ', np.round(dist_1mut[0,0], 4) )
                print('    attained with the three parameters:  ', np.around(dist_1mut[1,:3], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_1mut[2,:], 4) )
                print('    to be compared with real summary:    ',  np.around(stat_data, 4) )
                print("    Computing time: ", time_end-time_start)
                print()
            # Estimating the distance to model 2
            time_start = time()
            dist_2mut = dist_to_mod(2, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para2)
            time_end = time()
            if essai   == 0 :
                print()
                print('Distance to model 2: ', round(dist_2mut[0,0], 4) )
                print('    attained with the 4 parameters:      ', np.around(dist_2mut[1,:4], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_2mut[2,:6], 4) )
                print('    to be compared with real summary:    ',  np.around(stat_data, 4) )
                print("    Computing time: ", time_end-time_start)
                print()
            if dist_0mut[0,0] < dist_1mut[0,0] and dist_0mut[0,0] < dist_2mut[0,0]:
                confus[2,0]+= 1
            if dist_1mut[0,0] < dist_2mut[0,0] and dist_1mut[0,0] < dist_0mut[0,0]:
                confus[2,1]+= 1
            if dist_2mut[0,0] < dist_1mut[0,0] and dist_2mut[0,0] < dist_0mut[0,0]:
                confus[2,2]+= 1
                cumul_dist_stat_2mut += np.sum(np.absolute(1-(dist_2mut[2,:]  / stat_data)))
                cumul_dist_para_2mut += np.sum(np.absolute(1-(dist_2mut[1,:4] / para_data)))
            if essai != 0 and essai % 10 == 0:
                print()
                print('**************')
                print('Number of one mutation dataset treated  : ', essai+1)
                print('Accuracy for these datasets : ', confus[2,2] / (essai+1))
                print()
        
        print()
        print('********************************************************')
        print(' Conclusion : accuracy for double mutations data   = ', np.round(confus[2,2] * 100/numb_essai),' % ')
        if confus[2,2] != 0:
            print('         Mean relative error for parameter estimation : ', 
                      np.round(cumul_dist_para_2mut * 100/ (6 * confus[2,2]), 1),' % ' )
            print('         Mean relative error on statistical summary :   ', 
                      np.round(cumul_dist_stat_2mut * 100 / (8 * confus[2,2]), 1),' % ')
        print('********************************************************')
        
        #*****************************
        # GENERATION OF DATASET WITH THE NO MUTATION MODEL
        #****************************
        cumul_dist_para_0mut = 0
        cumul_dist_stat_0mut = 0
        #*****************************
        for essai in range(numb_essai):
            
            para_data = np.zeros(2, dtype=np.float64)
            alea_data = np.random.rand(2)
            para_data[0] = ampl_para0[0,0] + alea_data[0] * (ampl_para0[0,1]-ampl_para0[0,0])
            para_data[1] = ampl_para0[1,0] + alea_data[1] * (ampl_para0[1,1]-ampl_para0[1,0])
            
            stat_data = summ_stat(0, arbre, para_data, min_K, numb_data, age_declar_norm, age_declar_mut)
            if essai  == 0:
                print()
                print('**********************************************************************************')
                print('Generating a dataset in the no mutation model')
                print('    with parameters : ', np.round(para_data, 4) )
                print('    and statistical summary : ', np.round(stat_data , 4) )
                print()
                
            
            # Estimating the distance to model 0
            time_start = time()
            dist_0mut = dist_to_mod(0, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para0)
            time_end = time()
            if essai   == 0 :
                
                print()
                print('Distance to model 0 : ', np.round(dist_0mut[0,0], 4) )
                print('    attained with the two parameters    :', np.around(dist_0mut[1,:2], 4) )
                print('    to be compared with real parameters :', np.around(para_data, 4) )
                print('    giving the summary :                 ', np.around(dist_0mut[2,:], 4) )
                print('    to be compared with real summary :   ',  np.around(stat_data, 4) )
                print("    Computing time:  ", time_end-time_start)
                print()
            # Estimating the distance to model 1
            time_start = time()
            dist_1mut = dist_to_mod(1, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para1)
            time_end = time()
            if essai   == 0 :
                
                print()
                print('Distance to model 1 : ', np.round(dist_1mut[0,0], 4) )
                print('    attained with the three parameters:   ', np.around(dist_1mut[1,:3], 4) )
                print('    to be compared with real parameters:  ', np.around(para_data, 4) )
                print('    giving the summary:                   ', np.around(dist_1mut[2,:], 4) )
                print('    to be compared with real summary:     ',  np.around(stat_data, 4) )
                print("    Computing time: ", time_end-time_start)
                print()
            
            # eEstimating the distance to model 2
            time_start = time()
            dist_2mut = dist_to_mod( 2, stat_data , arbre, min_K, age_declar_norm, age_declar_mut, ampl_para2)
            time_end = time()
            if essai   == 0 :
                
                print()
                print('Distance to model 2 : ', round(dist_2mut[0,0], 4) )
                print('    attained with the 4 parameters:      ', np.around(dist_2mut[1,:4], 4) )
                print('    to be compared with real parameters: ', np.around(para_data, 4) )
                print('    giving the summary:                  ', np.around(dist_2mut[2,:6], 4) )
                print('    to be compared with real summary:    ',  np.around(stat_data, 4) )
                print("    Computing time:  ", time_end-time_start)
                print()
            if dist_1mut[0,0] < dist_2mut[0,0] and dist_1mut[0,0] < dist_0mut[0,0]:
                confus[0,1]+= 1
            if dist_2mut[0,0] < dist_1mut[0,0] and dist_2mut[0,0] < dist_0mut[0,0]:
                confus[0,2]+= 1
            if dist_0mut[0,0] < dist_1mut[0,0] and dist_0mut[0,0] < dist_2mut[0,0]:
                confus[0,0]+= 1
                cumul_dist_stat_0mut += np.sum(np.absolute(1-(dist_0mut[2,:] / stat_data)))
                cumul_dist_para_0mut += np.sum(np.absolute(1-(dist_0mut[1,:2]/para_data)))
            if essai != 0 and essai % 10   == 0:
                print('**************')
                print('Number of no mutation datasets treated: ', essai + 1)
                print('Accuracy for these datsets: ', confus[0,0] / (essai + 1))
                print()
        print()
        print('********************************************************')
        print('Conclusion : accuracy for no mutation data  = ', np.round(confus[0,0] * 100/numb_essai),' % ')
        if confus[0,0] != 0:
            print('       Mean relative error for parameter estimation :  ', 
                      np.round(cumul_dist_para_0mut * 100/ (2 * confus[0,0]), 1),' % ' )
            print('       Mean relative error on statistical summary :    ', 
                np.round(cumul_dist_stat_0mut * 100 / (8 * confus[0,0]), 1),' % ')
        print('********************************************************')

        print()
        print('********************************************************')
        print("Overall conclusion : confusion matrix")
        print("                          with ", numb_gener, " generations")
        print("                           and ", numb_kids, " children per couple")
        print()
        print(confus)
        print()
        print('********************************************************')
        
        
     
