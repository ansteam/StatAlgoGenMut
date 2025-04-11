
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
#**                ACTION ON REAL DATA                   **
#**               BREAST/OVARIAN CANCER                  **
#**********************************************************

# This software uses the distance minimization algorithm
# to analyse data on breast/ovarian cancer.
# The data is divided into 3 groups:
#     -a group of families where at least one woman in the family 
#                 has been tested positive on the BRCA mutation
#     -a group of families where no positive BRCA test has been obtained 
#                 but oncogeneticists suspect an hereditary cause
#     -a group of families where no genetic predisposition is suspected by 
#                 oncogeneticists

# For each group the software computes statistics summarizing the population
# which we call statitical summary.
# Then it determines between three possible choices of (parametrized) models
# the one that can give the closest statistical summary.

# The available data consists in the phenotypes of the individuals.
# The phenotype of an individual is the age of onset of the disease K 
# or 200 which mean by convention that K has not occured.
# Genotypes are  hidden but generated in the three models.
# In model 1, the genotype is 1 (mutated) or 0 (wild type) i
# In model 2, the genotype is [0,0] i.e. wild type or [1,0] i.e. having mutation 0
# or [0,1] i.e. having mutation 1  or [1,1] i.e. doubly mutated
# In model 0, no mutation is considered


# used packages
from time import time
import numpy as np
from numpy.random import rand
import numba as nb
import matplotlib.pyplot as plt
import realdata as rd

#****************************************
# Specifying the parameters in each model
#****************************************
# The distribution of the age of onset can follow two fixed distribution
# one for mutated (in model 1) or doubly mutated (in model 2) individuals
# the other one for the other cases: wild type or only one mutation in model 2
# In model 0 the distribution of the age of onset is a mixture of these two laws 
# These two distributions are respectively defined below

decla_mut =[0.0000001]*10+[0.01]*10+[0.02,0.04,0.08,0.15,0.3]+[0.5,0.7,0.9,1.2]+[1.4,1.8,2,2.3,2.5]+[2.7, 2.9, 3.1, 3.3, 3.5]
decla_mut += [3.6, 3.85, 3.9, 3.85, 3.6]+[3.5,3.4,3.3,3.1,3,2.9,2.8,2.7,2.6,2.5]+[2.4,2.2,2,1.8,1.6]
decla_mut += [1.4,1.2,1, 0.8, 0.75] +[0.7, 0.65, 0.6, 0.55,0.5]+[0.45,0.4,0.35,0.3,0.28]+[0.27,0.25,0.23,0.20,0.17]
decla_mut += [0.15,0.13,0.11,0.09,0.07]+ [0.06, 0.05, 0.04,0.03,0.025]+[0.01]*10+[0.005]
age_declar_mut = np.array([decla_mut[k]/100 for k in range(100)] , dtype=np.float64)

decla_norm = [0.0000001]*10+[0.01]*10+ [0.02]*5+ [0.04]*5+ [0.12]*5 +[0.2,0.4,0.5,0.6,0.7]+[0.8,0.9,1,1.1,1.2]
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

# The data is extracted from the Jean Perrin cancer center as an excel file.
# It is formatted by a purpose-made procedure "read-data" 
# which cleans the file, corrects errors, add missing values.
# The results is a 3 dimensional array of floats
# population[:,:,:], dtype=np.float64
# population[n, r, c] represents a person where
# n - family number
# r (0..7) row, different type of informations
# c (0..200) - column, information concerning person number c 
# First row (r=0) :
#   c = 0: size of the tree (number of members of the family)
#   c = 1: number of cancers in the family
#   c = 2: family identifier in the original file
# r = 1: (second row) identifier of corresponding father (-1 if not in the tree)
# r = 2: identifier of corresponding modther (-1 if not in the tree)
# r = 3: number of the generation, 0 is the most recent
# r = 4: sex (0=M, 1=F)
# r = 5: year of birth
# r = 6: age of cancer (200 for no cancer)

# The data and and the "read_data" procedure are not available for public use.
# But the software below can work on any set of real data as long as it is
# formatted as a Python array which has the structure described.



# ********************************************************************
# Generation of the genotype of a family in model 1 
# with a genealogical tree given by "arbr"
# with parameters given by "param"
@nb.jit(nb.int64[:,:](nb.int64[:,:], nb.float64[:]), nopython=True)
def draw_geno1(arbr , param):
    # rename more explicitely the variables contained in "arbr" or "param"
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
# The result is a one dimensionnal array 
# giving the age of onset of K if it has occured and 200 otherwise.
# the array "arbr" contains in particular
# the birth year as line 5
# the sex as line 4 (1 for female, 0 for male)
@nb.jit(nb.int64[:](nb.int64, nb.int64[:,:], nb.int64[:,:], nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
def phenotype(model, arbr, geno, param, adec_norm, adec_mut):
    curr_year = np.int64(2019)
    taill = arbr[0,0]
    sex = arbr[4,:]
    age = 99 * np.ones(taill, dtype=np.int64)
    for i in range(taill):
        if arbr[5,i] > curr_year - 99:
            age[i] = curr_year - arbr[5,i]
    adec0 = param[0] * adec_norm + (1-param[0]) * adec_mut
    adec1 = np.vstack((adec_norm, adec_mut))
    adec2 = np.vstack((adec_norm, adec_norm, adec_mut))
    if model == 2 :
        pene2 = np.array([param[2], param[2], param[2], param[3]]).reshape(2,2)
    phenoty = np.zeros(taill, dtype=np.int64)
    for i in range(taill):
        if sex[i] == 1:
            if model == 0:
                normali = np.sum(adec0[ 0:age[i]])
                proba_K = min (param[1] * normali, 0.9999)
                vect_proba = adec0[ 0:age[i]] / normali
            if model == 1:
                normali = np.sum(adec1[geno[i,0], 0:age[i]])
                proba_K = min (param[geno[i,0]+1] * normali, 0.9999)
                vect_proba = adec1[geno[i,0], 0:age[i]] / normali
            if model == 2:
                sum_geno = np.sum(geno[i, :])
                normali = np.sum(adec2[sum_geno, 0:age[i]])
                vect_proba = adec2[sum_geno, 0:age[i]] / normali 
                proba_K = min(pene2[geno[i,0], geno[i,1]] * normali, 0.9999)
            if rand() > proba_K :            
                phenoty[i] = 200
            else:
                ff = np.cumsum(vect_proba)
                ff[-1] = 1
                j = 0
                ch = rand()
                while ch > ff[j]:
                    j += 1
                phenoty[i] = j
    return phenoty








#*******************************************************************
# Notion of proximity between to individual phenotypes
# being both  1x2 arrays 
# The possible results are -3, 1, 5, 10
# meaning increasing proximities
@nb.jit(nb.int64( nb.int64, nb.int64), nopython = True)
def proxi(phen1, phen2):
    if phen1 == 200 and phen2 == 200 :
        return  0    
    if phen1 < 50 and phen1 != -1 and phen2 == 200 :
        return -1
    if phen1 == 200 and phen2 < 50 and phen2 != -1:
        return -1
    if (phen1 >= 50 or phen1 == -1) and phen2 == 200 :
        return 0
    if phen1 == 200 and (phen2 >= 50 or phen2 == -1) :
        return 0
    if phen1 < 200 and phen1 != -1 and phen2 < 200 and phen2 != -1 :
        if phen1 < 50 and phen2 < 50:
            return 10 
        else:
            if abs(phen1 - phen2) < 11:
                return 6  
            else:
                return 3
    return 0


#****************************************************************
# Computing one of the statistics under consideration,
# a proximity score between a woman and her mother and grandmothers
@nb.jit(nb.float64( nb.int64[:,:], nb.int64[:]), nopython=True)
def score_par(arbr, pheno):
    taill = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    sex = arbr[4,:]
    scor_par = 0
    numb_par_gdpar = 0
    for ind in range(taill):
        if sex[ind] == 1:
            #if this woman has a mother (in the tree)
            if moth[ind] != -1:
                numb_par_gdpar += 1
                scor_par +=   proxi(pheno[ind], pheno[moth[ind]])
                #if this mother has a mother (in the tree)
                if moth[moth[ind]] != -1:
                    scor_par +=  proxi(pheno[ind], pheno[moth[moth[ind]]])
                    numb_par_gdpar += 1
            # also look at the grandmother on the father side
            if fath[ind] != -1:
                if moth[fath[ind]] != -1:
                    scor_par +=  proxi(pheno[ind], pheno[moth[fath[ind]]])
                    numb_par_gdpar += 1
    if numb_par_gdpar == 0:
        return 0
    else:
        return scor_par /  numb_par_gdpar

#****************************************************************
# Computing another statistics under consideration,
# a proximity score between a woman and her sisters
@nb.jit(nb.float64( nb.int64[:,:], nb.int64[:]), nopython=True)
def score_sist(arbr, pheno):
    # renommer les variables en clair 
    taill = arbr[0,0]
    fath = arbr[1,:]
    moth = arbr[2,:]
    sex = arbr[4,:]
    scor_sist = 0
    numb_tot_sis = 0
    for ind in range(taill):
        if sex[ind] == 1:
            sisters = np.zeros(taill, dtype=np.int64)
            sis_count = 0
            # si l'individu a un père
            if fath[ind] != -1 or moth[ind] != -1 and ind < taill - 1 :
                # on scanne tous les individus d'indice plus grands
                for pot_sis in range(ind+1, taill):
                    if fath[pot_sis] == fath[ind] and moth[pot_sis] == moth[ind]:
                        sisters[sis_count] = pot_sis
                        sis_count += 1
                        numb_tot_sis += 1
            for sis in range(sis_count):
                scor_sist +=  proxi(pheno[ind], pheno[sisters[sis]])
    if numb_tot_sis ==0:
        return 0
    else:
        return scor_sist / numb_tot_sis
    
    
# *****************************************************************
# Function computing the statistical summary of the real data "fdata"
# See above for the sructure of "fdata"
# The result is a one dimensional array containing:
# Stat 0: frequence of K among women
# Stat 1: frequence of K among the mothers in the tree  of  women who have K
# Stat 2: mean correlation score with female ascendants
# Stat 3: mean correlation with sisters
# Stat 4 to 23 : distribution of the age of onset with 5 years unit
# Stat 24 : set to 0 (used only for simulated data)
@nb.jit(nb.float64[:]( nb.int64[:,:,:]), nopython=True)
def desc_stat( fdata ):
    nb_arbr = len(fdata[:,0,0])
    result = np.zeros(25,  dtype=np.float64)
    numb_K = 0
    numb_W = 0
    numb_K_w_moth = 0
    numb_K_for_mothK = 0
    loc_scor_par = 0
    loc_scor_sist = 0
    for ia in range(nb_arbr):
        taill = fdata[ia,0,0]
        curr_arbr = fdata[ia, : , :taill ]
        moth = curr_arbr[2,:]
        sex = curr_arbr[4,:]
        pheno = fdata[ia, 6, :taill]
        for indi in range(taill):
            if sex[indi] == 1 :
                numb_W += 1
                if pheno[indi] < 100 and pheno[indi] > 1:
                    numb_K += 1
                    result[4 + pheno[indi] // 5 ] += 1
                    if moth[indi] == -1 :
                        numb_K_w_moth += 1
                        if pheno[moth[indi]] < 100 and pheno[moth[indi]] > 1:
                            numb_K_for_mothK += 1
                
        loc_scor_par += score_par(curr_arbr, pheno)
        loc_scor_sist += score_sist(curr_arbr, pheno)
    # restitution des résultats 
    if numb_W != 0:
        result[0] = numb_K / numb_W
    if numb_K_w_moth !=0:
        result[1] = numb_K_for_mothK / numb_K_w_moth 
    result[4:24] = result[4:24] / np.sum(result[4:24])
    result[2] = loc_scor_par / nb_arbr
    result[3] = loc_scor_sist / nb_arbr
    return result

#***********************************************
# This function computes the statistical summary 
# in model 0 or 1 or 2
# for a given set of parameters given by "param"
# for a genealogical structure given by "genea"
# where the number of K for each family tree is prescribed.
# This is done by simulation of "loc_repliq" replica 
@nb.jit(nb.float64[:](nb.int64, nb.int64[:,:,:], nb.float64[:], nb.int64, nb.float64[:], nb.float64[:]), nopython=True)
def summ_stat(model, genea, param, loc_repliq, adec_norm, adec_mut):
    nb_arbr = len(genea[:,0,0])
    #print("appel summ<-stat, modèle=",model,"param=",param)
    # le nombre de tirages pour le calcul de ces stats
    # est loc_numb_simu 
    # initialisation desrésultats
    result = np.zeros(25,  dtype=np.float64)
    numb_W =0
    numb_K = 0
    numb_K_w_moth = 0
    numb_K_for_mothK = 0
    loc_scor_par = 0
    loc_scor_sist = 0
    enough = False
    count_sim = 0
    compt_reduc_K = 0
    while (count_sim < loc_repliq or enough == False):
        for ia in range(nb_arbr):
            curr_arbr = genea[ia,:,:]
            taill = curr_arbr[0,0]
            nb_K = curr_arbr[0,1]
            moth = curr_arbr[2,:]
            sex = curr_arbr[4,:]
            number_K = -1
            compt_infruc = np.int64(0)
            # on va tirer jusqu'à ce que le phénotype
            # contienne au moins loc_min_K cas de K
            while number_K != nb_K :
                if model == 0:
                    loc_geno = np.zeros((taill, 2), dtype=np.int64)
                if model == 1:
                    loc_geno = draw_geno1(curr_arbr, param)
                if model == 2:
                    loc_geno = draw_geno2(curr_arbr, param)
                loc_pheno = phenotype(model, curr_arbr, loc_geno, param, adec_norm, adec_mut)
                #compte le nombre de K
                number_K = taill - np.sum(loc_pheno[:] == 200)- np.sum(loc_pheno[:] == 0)
                compt_infruc += 1
                if compt_infruc == 5 *  curr_arbr[0,1] and number_K < nb_K:
                    nb_K -= 1
                    compt_reduc_K += 1
                    compt_infruc = 0
                if compt_infruc == 5 *  curr_arbr[0,1] and number_K > nb_K:
                    nb_K += 1
                    compt_reduc_K += 1
                    compt_infruc = 0
            # on sort du while précédent avec un tirage convenable
            # il n'y a plus qu'à calculer les stats
            for indi in range(taill):
                if sex[indi] == 1 :
                    numb_W += 1
                    if loc_pheno[indi] < 100 and loc_pheno[indi] > 1:
                        numb_K += 1
                        result[4 + loc_pheno[indi] // 5 ] += 1
                    if moth[indi] == -1 :
                        numb_K_w_moth += 1
                        if loc_pheno[moth[indi]] < 100 and loc_pheno[moth[indi]] > 1:
                            numb_K_for_mothK += 1
                
            loc_scor_par += score_par(curr_arbr, loc_pheno)
            loc_scor_sist += score_sist(curr_arbr, loc_pheno)
            
        if  numb_K_w_moth >= 1000 :
                enough = True
        count_sim += 1
    result[0] = numb_K / numb_W
    result[1] = numb_K_for_mothK / numb_K_w_moth
    result[4:24] = result[4:24] / np.sum(result[4:24])
    result[2] = loc_scor_par / (nb_arbr*count_sim)
    result[3] = loc_scor_sist / (nb_arbr*count_sim)
    #result[24] = compt_reduc_K / (nb_arbr * loc_repliq )
    #print(result)
    return result



#*************************************************************************
# Computing the distance to each of the models 
# and obtaining th best parameters 
# for a dataset completely summarized by its summary "sta_summ"
# The algoritm below seeks the set of  parameters 
# for which statistical summary is the closest to "sta_summ"
# Parameters of this function have the same meaning as usual
# "ampl_para" defines in what intervals the optimal values of parameters are searched
@nb.jit(nb.float64[:,:]( nb.int64, nb.float64[:], nb.int64[:,:,:], nb.float64[:], nb.float64[:], nb.float64[:,:]), nopython=True)
def dist_to_mod(model, sta_summ, genea, adec_norm, adec_mut, ampl_para):
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
    
    #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # OPERATING PARAMETERS TO BE TUNED
    # Choose the number of points to form a grid
    nbr_maill_para =  np.ones( dim_para, dtype = np.int64) * 10
    # Setting the number of simulation in the first and second phases
    loc_repliq_ini = np.int64(1)
    loc_replic_refin = np.int64(5)
    # Setting the number of kept values in the first phase
    size_best = 10
    # During second phase, number of tries around the previously selected value
    loc_try_refin = 10
    #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    
    # First search the minimum on a grid
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
        curr_stat =   summ_stat(model , genea, curr_para, loc_repliq_ini, adec_norm, adec_mut)
        # Distance computation
        curr_dist = np.float64(0)
        for i in range(dim_summ):
            if curr_stat[i] != 0 and sta_summ[i] != 0:
                curr_dist += coeff[i] * ( max(curr_stat[i] / sta_summ[i], sta_summ[i] / curr_stat[i] ) - 1 ) ** 2
            else:
                curr_dist += coeff[i] * ( curr_stat[i] - sta_summ[i]) ** 2
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
            refin_stat =  summ_stat(model, genea, refin_para, loc_replic_refin, adec_norm, adec_mut)
            #Distance evaluation
            refin_dist = np.float64(0)
            for i in range(dim_summ):
                if refin_stat[i] != 0 and sta_summ[i] != 0:
                    refin_dist += coeff[i] * ( max(refin_stat[i] / sta_summ[i], sta_summ[i] /refin_stat[i] ) - 1 ) ** 2
                else:
                    refin_dist += coeff[i] * ( refin_stat[i] - sta_summ[i]) ** 2
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
# Randomness initialization 
    
np.random.seed(33)


print()
print("**************************************************")
print("*   MODEL SELECTION AND PARAMETRIC ESTIMATION    *")
print("*   USING THE DISTANCE MINIMIZATION ALGORITHM    *")
print("*          FOR REAL DATA CONCERNING              *")
print("*          THE BREAST OVARIAN CANCER             *")
print("*                                                *")
print("* Copyright : F. Kwiatkowski, L. Serlet, A. Stos *")
print("**************************************************")
print()
print()

print()
print("Three models are under scrutiny:")
print("     Model 0 : no  mutation and 2 parameters")
print("     Model 1 : one mutation and 3 parameters")
print("     Model 2 : two mutations and 4 parameters")
print()

#****************************************
# importing real data
#population = rd.read_data('Data/All_SO.xls')
#population = rd.read_data('Data/SO_mut.xls')
#population = rd.read_data('Data/SO_no_pred.xls')
population = rd.read_data('Data/SO_pred_no_mut.xls')

print()
print("fichier étudié : SO_pred_no_mut.xls")

# Data structure: 3 dimensional array of floats
# population[:,:,:], dtype=np.float64
# population[n, r, c] represents a person where
# n - family number
# r (0..7) row, different type of informations
# c (0..200) - column, information concerning person number c 
# First row (r = 0) :
#   c = 0: size of the tree (number of members of the family)
#   c = 1: number of cancers in the family
#   c = 2: family identifier in the original file
# r = 1: (second row) identifier of corresponding father (-1 if not in the tree)
# r = 2: identifier of corresponding modther (-1 if not in the tree)
# r = 3: number of the generation, 0 is the most recent
# r = 4: sex (0=M, 1=F)
# r = 5: year of birth
# r = 6: age of cancer (or 200 for no cancer)

# Notation for the genealogical structure 
d_genea = population[:,:6,:]
# Statistical summary for real data
r_stat = desc_stat(population)
#print("r_stat = ", r_stat)
print()
plt.plot(np.linspace(0,100,20), r_stat[ 4:24], color = 'blue', label='data')


# distance to 0 mutation model
time_start = time()
dist_0mut = dist_to_mod(0,r_stat ,d_genea, age_declar_norm, age_declar_mut, ampl_para0)
time_end = time()
print()
print('Distance to the no mutation model: ', dist_0mut[0,0] )
print('    reached for the two parameters: ', dist_0mut[1,:2] )
print('    which give the following summary: ', dist_0mut[2,:4] )
print('    to be compared with the real one: ',  r_stat[:4])
print('    and the value of the difficulty of simulation: ', dist_0mut[2,24])
plt.plot(np.linspace(0,100,20), dist_0mut[2,4:24], color = 'orange', label='0 mut.')        
print()
print("    Computing time: ", time_end-time_start)
print()

# distance to 1 mutation model
time_start = time()
dist_1mut = dist_to_mod(1,r_stat ,  d_genea, age_declar_norm, age_declar_mut, ampl_para1)
time_end = time()

print()
print('Distance to the one mutation model: ', dist_1mut[0,0] )
print('    reached for the three parameters: ', dist_1mut[1,:3] )
print('    which give the following summary: ', dist_1mut[2,:4] )
print('    to be compared with the real one: ',  r_stat[:4])
print('    and the value of the difficulty of simulation: ',  dist_1mut[2,24])
plt.plot(np.linspace(0,100,20), dist_1mut[2,4:24], color = 'red', label='1 mut.')        
print()
print("    Computing time: ", time_end-time_start)
print()


# distance to 2 mutations model
time_start = time()
dist_2mut = dist_to_mod(2, r_stat ,  d_genea, age_declar_norm, age_declar_mut, ampl_para2)
time_end = time()

print()
print('Distance to the two mutations model: ', dist_2mut[0,0] )
print('    reached for the four parameters: ', dist_2mut[1,:4] )
print('    which give the following summary: ', dist_2mut[2,:4] )
print('    to be compared with the real one: ',  r_stat[:4])
print('    and the value of the difficulty of simulation: ',  dist_2mut[2,24])
print()
print("    Computing time: ", time_end-time_start)
print()
plt.plot(np.linspace(0,100,20), dist_2mut[2,4:24], color = 'maroon', label='2 mut.')


plt.title("Distribution of the age of cancer :  data  vs the 3 fitted models")
plt.legend()
plt.show()
    


