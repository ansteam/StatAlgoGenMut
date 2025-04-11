import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time 
from colorama import Fore, Back, Style

# Common header (15 columns) in all files
# ['N° famille' 'Prédispo. familiale' 'Famille mutée' 'Gène muté'
#  'N° membre' 'Sexe' 'Individu muté' 'N° père' 'N° mère' 'Année naissance'
#  'N° géné- ration' 'Age cancer' 'Localisation cancer' 'Décès' 'Age décès']

# Output data structure: 3 dimensional array of floats
# population[:,:,:], dtype=np.float64
# population[n, r, c] represents a person where
# n - family number
# r (0..7) row, different type of informations
# c (0..200) - column, information concerning person number c 
# First row (r=0) :
# c = 0: size of the tree (number of members of the family)
# c = 1: number of cancers in the family
# c = 2: family identifier in the original file
# r = 1: (second row) identifier of corresponding father (-1 if not in the tree)
# r = 2: identifier of corresponding modther (-1 if not in the tree)
# r = 3: number of the generation, 0 is the most recent
# r = 4: sex (0=M, 1=F)
# r = 5: year of birth
# r = 6: age of cancer (200 for no cancer)

# CONSTANTS for population array
row_info = 0
row_fath = 1
row_moth = 2
row_gener = 3
row_sex = 4
row_yofb = 5
row_age = 6
info_family_size = 0
info_nb_cancer = 1
info_family_id = 2

# Random seed for data augmentation (setting unknown sex to random)
np.random.seed(87654)
np.set_printoptions(suppress=True) # do not use scientific notation for arrays


def get_family_size(population, t_ind):
    return int(population[t_ind, row_info, info_family_size])


def get_population_size(population):
    return population.shape[0]


def get_nb_cancers(population, f_ind):

    return int(population[f_ind, row_info, info_nb_cancer])


def get_family_id(population, f_ind):
    # Given family index in the population, return the family ID in the source file
    # (as stored in population array)
    return int(population[f_ind, row_info, info_family_id])


def find_family_population_index(population, family_id):
    # Find the index of the family in popuulation given its ID in the source file
    nb_trees = get_population_size(population)
    for i in range(nb_trees):
        if population[i, row_info, info_family_id] == family_id:
            return i    


def display(population, f_ind):
    # Displays family f_ind from the population
    print(f"Tree number: {f_ind}, family ID in the source file : {population[f_ind, 0, 2]:.0f}" )
    # family_size = int(population[f_ind, 0, 0])
    family_size = get_family_size(population, f_ind)
    family = population[f_ind, :, :]
    maxgen = np.max(population[f_ind, 3, 0:family_size])
    curr_gen = maxgen
    print(f'    GEN {int(curr_gen)}:')
    for mem in range(family_size):
        if family[row_gener, mem] != curr_gen:
            curr_gen = family[row_gener, mem]
            print(f'\n    GEN {int(curr_gen)}:')
        yofb_style = ""
        if np.isnan(family[row_yofb, mem]):
            yofb_style = Back.LIGHTYELLOW_EX+Fore.YELLOW
        sex_style = Fore.LIGHTBLUE_EX
        if family[row_sex, mem] == 1:
            sex_style = Fore.LIGHTRED_EX
        illness = "."
        cancer_style = ""
        if family[row_age, mem] < 200:
            illness = int(family[row_age, mem])
            cancer_style = Fore.RED
        print( f"({sex_style}{mem}{Style.RESET_ALL}, {family[row_fath, mem]:.0f}, {family[row_moth, mem]:.0f}, {yofb_style}{family[row_yofb, mem]:3.0f}{Style.RESET_ALL}, {cancer_style}{illness}{Style.RESET_ALL})  ", end='')
    print()
    print('-'*80)


def augment_yofb_generation(ind, population):
    family_size = int(population[ind, 0, 0])
    family = population[ind, :, 0:family_size]
    # Computing average year of birth for all generations
    maxgen = int(np.max(family[row_gener, 0:family_size]))
    curr_gen = maxgen
    curr_sum = 0
    curr_N = 0
    gen_averages = [0] * int(maxgen+1)
    for mem in range(family_size):
        if family[row_gener, mem] != curr_gen:
            # Next generation detected
            if curr_N > 0:
                # Compute the average for the previous generation
                gen_averages[curr_gen] = np.round(curr_sum / curr_N)
                curr_sum = 0
                curr_N = 0
                curr_gen = int(family[row_gener, mem])
        if not np.isnan(family[row_yofb, mem]):
            curr_sum += family[row_yofb, mem]
            curr_N += 1
    # Average for generation 0
    if curr_N > 0:
        gen_averages[curr_gen] = np.round(curr_sum / curr_N)
        curr_sum = 0
        curr_N = 0
        curr_gen = int(family[row_gener, mem])
    unresolved = 0
    for i in range(family_size):
        if np.isnan(family[row_yofb, i]):
            gener = int(family[row_gener, i])
            if gen_averages[gener] > 0:
                family[row_yofb, i] = gen_averages[gener]
            else:
                unresolved += 1
    return unresolved


def augment_yofb_parents_children(ind, population):
    family_size = int(population[ind, 0, 0])
    # For each person with unknown year of birth, look for their parents
    both_parent_undef_count = 0
    for i in range(family_size):
        if np.isnan(population[ind, row_yofb, i]):
            if population[ind, row_moth, i] == -1 and population[ind, row_fath, i] == -1:
                both_parent_undef_count += 1
            else:
                m = int(population[ind, row_moth, i])
                if m > -1:
                    # Mother is found, take her year of birth as a base
                    y_m = population[ind, row_yofb, m]
                    birth_y = y_m + 25
                else:
                    # Father is in the family (mother wasn't)
                    f = int(population[ind, row_fath, i])
                    # father's year of birth
                    y_f = population[ind, row_yofb, f] 
                    birth_y = y_f + 25
                population[ind, row_yofb, i] =  birth_y
    # If there are still unresolved cases, try looking for children    
    unresolved = 0
    for i in range(family_size):
        if np.isnan(population[ind, row_yofb, i]):           
            # Check if the member is father of someone in the family
            locate_children_f = np.where(population[ind, row_fath, 0:family_size] == i)
            # Check if the member is mother of someone in the family
            locate_children_m = np.where(population[ind, row_moth, 0:family_size] == i)
            if np.size(locate_children_f) + np.size(locate_children_m) > 0:
                if np.size(locate_children_f) > 0 and np.size(locate_children_m) > 0:
                    # This should never happen, except for bugs.
                    print('Father is Mother too. Ooops.')
                    sys.exit(0)
                if np.size(locate_children_f) > 0:
                    # This member is a father
                    if np.all(np.isnan(population[ind, row_yofb, locate_children_f])):
                        unresolved += 1
                    else:
                        first_child_yofb = np.nanmin(population[ind, row_yofb, locate_children_f])
                        population[ind, row_yofb, i] = first_child_yofb - 25
                else:
                    # If member is not a father, it must be a mother
                    if np.all(np.isnan(population[ind, row_yofb, locate_children_m])):
                        unresolved += 1
                    else:
                        first_child_yofb = np.nanmin(population[ind, 5, locate_children_m])
                        population[ind, row_yofb, i] = first_child_yofb - 25
            else:
                unresolved += 1
    return unresolved


def read_data(filename):
    dat_given = pd.read_excel(filename, header=3)
    dat_noms = [filename]
    datlist = [dat_given]

    # Cancer types retained for analysis 
    retained_types = ['sein', 'ovai', 'gyn', 'vagin', 'vulve', 'utérus', 'fibrome', 'uterus', 'incon', 'NR']

    # Searching for the biggest family tree.
    biggest_tree = 0
    current_family_id = 0
    member_count = 0
    family_sizes = []
    family_ids = []
    for nom, dat in zip(dat_noms, datlist):
        nbr_entries = dat.shape[0]
        print(filename, nbr_entries, "entries (persons)")
        for i in range(nbr_entries):
            # There are some empty lines in one file
            if str(dat.iloc[i,0]) == 'nan':
                # Missing family number, skipping (there are empty lines)
                continue
            # Are there non-numeric family numbers? (just in case)
            if not str(dat.iloc[i, 0]).replace('.','').isdigit():
                print("Mising data in", nom, "line", i, "entry", str(dat.iloc[i, 0]))
                sys.exit(1)
            if int(dat.iloc[i, 0]) != current_family_id:
                # print("Family", current_family_id, "has", member_count, "members")
                if i > 0: #
                    family_sizes.append(member_count)
                    family_ids.append(current_family_id)
                current_family_id = int(dat.iloc[i, 0])
                if member_count > biggest_tree:
                    biggest_tree = member_count
                member_count = 0
            member_count += 1
    family_count = len(family_sizes)
    print("Found", family_count, 'families')
    print("Biggest family tree has", biggest_tree, 'members')
    print()
    # Histogram
    # plt.hist(family_sizes, bins=np.arange(0, 161, 2))
    # plt.show()

    # 200 is OK for now, but take max just in case there was a new dataset
    biggest_tree = max(biggest_tree, 200)

    # Variable population contains all the family trees
    population = -99 * np.ones((family_count, 7, biggest_tree), dtype=np.float64)
    population_int = -99 * np.ones((family_count, 7, biggest_tree), dtype=np.int)    
    random_sex_count = 0
    retained_cases = []
    rejected_cases = []
    corrected_cancer_age = 0
    for tree_ind, fid in enumerate(family_ids):
        family = dat[dat['N° famille'] == fid]

        # tree[0,0] = Number of members of the family
        family_size = family.shape[0]
        population[tree_ind, 0, 0] = family_size

        # tree[0,1] = Number of ill members
        # nb_malad = family[family['Age cancer'] < 200].shape[0]
        # population[tree_ind, 0, 1] = nb_malad
        population[tree_ind, 0, 2] = fid

        # First check on basic data (another one on case-by-case basis below)
        population[tree_ind, row_age, :family_size] = family['Age cancer'].values
        if np.any(np.isnan(family['Age cancer'].values)):
            print("Missing cancer age.") # does not happen yet, keep just in case.
            sys.exit(0)

        # Re-indexing family members (0-based)
        member_ids = list(family['N° membre'].values)
        father_ids = list(family['N° père'].values)
        father_ids = list(map(int, father_ids))
        mother_ids = list(family['N° mère'].values)
        mother_ids = list(map(int, mother_ids))
        nb_malades = 0
        for m, mem_id in enumerate(member_ids):
            # Compute the parents
            father_id = father_ids[m]
            mother_id = mother_ids[m]
            if father_id == -1:
                # Father not in this family tree
                father = -1 
            else:
                # Father belongs in this family tree
                father = member_ids.index(father_id)
            if mother_id == -1:
                # Mother not in this family tree
                mother = -1
            else:
                # Mother belongs in this family tree
                mother = member_ids.index(mother_id)
            population[tree_ind, 1, m] = father
            population[tree_ind, 2, m] = mother
            # Convert sex to numerical: M=0, F=1
            if family.iat[m, 5] == 'F':
                population[tree_ind, row_sex, m] = 1
            if family.iat[m, 5] == 'M':
                population[tree_ind, row_sex, m] = 0
            # Testing for undefined values
            if family.iat[m, 5] not in ['M', 'F']:
                if int(mem_id) in father_ids:
                    print("Data processing: Sex undefined, set to M by looking at fathers")
                    population[tree_ind, row_sex, m] = 0
                elif int(mem_id) in mother_ids:
                    print("Data processing: Sex undefined, set fo F by looking at mothers")
                    population[tree_ind, row_sex, m] = 1
                elif 'sein' in str(family.iat[m, 12]) or 'gyn' in str(family.iat[m, 12]) or 'ova' in str(family.iat[m, 12]) or 'col' == str(family.iat[m, 12]):
                    print("Data processing: Sex undefined, set to F per cancer type")
                    population[tree_ind, row_sex, m] = 1
                elif 'prostate' in str(family.iat[m, 12]) or 'testic' in str(family.iat[m, 12]):
                    print("Data processing: Sex undefined, set to man per prostate cancer")
                    population[tree_ind, row_sex, m] = 0
                else: 
                    # Undefined sex, choosing at random
                    population[tree_ind, row_sex, m] = np.round(np.random.rand())
                    random_sex_count += 1
            # If cancer, verify the type, retain only interesting ones and unknowns.
            if population[tree_ind, row_age, m] < 200:
                retain = False
                for c_type in retained_types:
                    if c_type in str(family.iat[m, 12]) or str(family.iat[m, 12]) == 'col' :
                        if population[tree_ind, row_sex, m] != 0:
                            retain = True
                if retain:
                    nb_malades += 1
                    age = population[tree_ind, row_age, m]
                    # If age on cancer detection is unknown, make it 60, a neutral value
                    if age == 0:
                        population[tree_ind, row_age, m] = -1
                        corrected_cancer_age += 1
                    retained_cases.append(str(family.iat[m, 12]))
                else:
                    # Remove cancers of uninteresting type
                    population[tree_ind, row_age, m] = 200
                    rejected_cases.append(str(family.iat[m, 12]))
        population[tree_ind, row_info, info_nb_cancer] = nb_malades
        generations = family['N° géné- ration'].values
        maxgen = np.max(generations)
        population[tree_ind, 3, :family_size] = maxgen - generations
        if np.any(np.isnan(generations)):
            # It does not happen, but still.
            print("Missing generation.")
            sys.exit(0)
        population[tree_ind, row_yofb, :family_size] = family['Année naissance'].values
    print(f"Data processing: corrected age of cancer (0 -> -1) for {corrected_cancer_age} persons")
    print(f"Data processing: {random_sex_count} entries have undefined sex value (set to random)")
    print()
    print("Retained types:", len(retained_cases), '\n', set(retained_cases))
    print()
    print("Rejected types:", len(rejected_cases), '\n', set(rejected_cases))

    # Multipass parents-children analysis of the birth years
    unres_prev = 0
    for p in range(15):
        unres_current = 0
        unres_tree_count =  0
        nb_trees = len(family_ids)
        for t_ind in range(nb_trees):
            family_size = get_family_size(population, t_ind)
            if np.any(np.isnan(population[t_ind, row_yofb, 0:family_size])):
                unres_current += augment_yofb_parents_children(t_ind, population)
                unres_tree_count += 1
        print(f"After pass {p+1}, unknown year of birth for {unres_current} individuals (visited {unres_tree_count} family trees)")
        if unres_prev == unres_current:
            print("No more improvements, finishing parents-children analysis.\n")
            break
        unres_prev = unres_current
    
    # The left unknown years of birth set to the average of the generation
    unres = 0
    for t_ind in range(nb_trees):
        family_size = get_family_size(population, t_ind)
        if np.any(np.isnan(population[t_ind, row_yofb, 0:family_size])):
            unres += augment_yofb_generation(t_ind, population)
    print(f"After generation average pass, there is {unres} unresolved cases.")
    # Display unresolved cases
    for t_ind in range(nb_trees):
        family_size = get_family_size(population, t_ind)
        if np.any(np.isnan(population[t_ind, row_yofb, 0:family_size])):
            # display(population, t_ind)
            fid = get_family_id(population, t_ind)
            print(f"Family {t_ind}, source ID: {fid}, has unresolved year of birth problem.")
    prune_all_trees(population)
    # Checking for nans
    nans = False
    for t_ind in range(nb_trees):
        family_size = get_family_size(population, t_ind)
        if np.any(np.isnan(population[t_ind, row_yofb, 0:family_size])):
            print(t_ind)
            nans = True
    if nans:
        print("------------ STILL SOME UDEFINED VALUES. EXITING ---------------")
        sys.exit(0)
    population_int[:,:,:] = population[:, :, :]
    return population_int


def get_neighbors(population, family_ind, member):
    # Given the family number family_ind in the population,
    # find neighbors (children, parents) in the family tree for a given member.
    family_size = int(population[family_ind, 0, 0])
    family = population[family_ind, :, 0:family_size]
    neighbors = []
    # Find children
    if np.any(family[row_fath, 0:family_size] == member):
        children_mask = family[row_fath, 0:family_size] == member
        neighbors.extend(children_mask.nonzero()[0])
    if np.any(family[row_moth, 0:family_size] == member):
        children_mask = family[row_moth, 0:family_size] == member
        neighbors.extend(children_mask.nonzero()[0])
    # Append parents
    father = family[row_fath, member]
    mother = family[row_moth, member]
    if father > -1:
        neighbors.append(father)
    if mother > -1:
        neighbors.append(mother)
    neighbors = list(map(int, neighbors))
    return neighbors


def get_connected_component(population, family_index, member):
    # Return all family members connected with member by a parental link
    # The member is included in the list.
    pile = [member]
    visited = []
    while len(pile) > 0:
        current_member = pile.pop()
        if current_member not in visited:
            visited.append(current_member)
            pile.extend(get_neighbors(population, family_index, current_member))
    return visited


def get_largest_connected_component(population, family_index):
    # Return the largest connected component in the family tree
    # and the number of components for data analysis
    family_size = int(population[family_index, 0, 0])
    all_components = []
    visited = []
    for member in range(family_size):
        if member not in visited:
            component = get_connected_component(population, family_index, member)
            all_components.append(component)
            visited.extend(component)
    sizes = list(map(len, all_components))
    imax = np.argmax(sizes)
    return all_components[imax], len(all_components)


def prune_family_tree(population, family_index):
    # If the given family has more than one connected components,
    # delete the minor components, reindex the largest connected component.
    # Operation is made in-place (directly in the array population).
    # family_size = int(population[family_index, 0, 0])
    family_size = get_family_size(population, family_index)
    largest_comp, _ = get_largest_connected_component(population, family_index)
    largest_comp.sort()
    family = population[family_index, :, 0:family_size]
    old_maxgen = np.max(family[row_gener, 0:family_size])
    nb_cancers = 0
    # print("Prunning")
    # print(largest_comp)
    # print("Family tree before")
    # display(population, family_index)
    for i, mem in enumerate(largest_comp): 
        if family[row_age, mem] < 200:
            nb_cancers += 1
        father = int(family[row_fath, mem])
        mother = int(family[row_moth, mem])
        new_father = -1
        if father > -1:
            new_father = largest_comp.index(father)
        new_mother = -1
        if mother > -1:
            new_mother = largest_comp.index(mother)
        family[1:, i] = family[1:, mem]
        family[row_fath, i] = new_father
        family[row_moth, i] = new_mother
    # Reset the size
    family[row_info, 0] = len(largest_comp)
    family[:, family_size:] = -99
    # Reset the number of cancers
    family[row_info, 1] = nb_cancers
    # Recompute generations, assumong that there is no generatin gap
    new_maxgen = len(set(family[row_gener, :len(largest_comp)]))
    delta = old_maxgen - new_maxgen + 1
    for mem  in range(len(largest_comp)):
        family[row_gener, mem] = family[row_gener, mem] - delta


def prune_all_trees(population):
    # At present, most cleaning is done while reading
    # (augmentation for year of birth end sex).
    # Here we take care only of disonnected family trees (prunning).
    # It could be refactored to better separate fonctions.
    print("\nPrunning disconnected members")
    print("Cuts bigger than 5:")
    nb_families = get_population_size(population)
    cut_hist = {}
    for f_ind in range(nb_families):
        family_size = get_family_size(population, f_ind)    
        largest_compo, nb_conn_compos = get_largest_connected_component(population, f_ind)
        if nb_conn_compos > 1:
            cut = family_size - len(largest_compo)
            if cut in cut_hist:
                cut_hist[cut] += 1
            else:
                cut_hist[cut] = 1
            prune_family_tree(population, f_ind)
            if cut > 5:
                print(f"    Family {f_ind}, cut off {cut} members.")
    print("Histogram:")
    for k in sorted(cut_hist.keys()):
        print(f"    Families with {k} members cut:", cut_hist[k])

if __name__ == "__main__":
    # population = read_data('Data/Predisposition Aucune.xls')
    population = read_data('Data/Predisposition SEIN-OVAIRE non mute.xls')

    # Number of cancers analysis
    # nb_canc = population[:, row_info, info_nb_cancer]
    # plt.hist(nb_canc, bins=np.arange(-0.5, 15, 1))
    # plt.show()
            
    