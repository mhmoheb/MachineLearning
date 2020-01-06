# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:14:36 2019

@author: Maryam
"""
import itertools
import numpy
import pandas
import sklearn.tree as tree


  
# Question 1 - Part (A)
    
Customer_survey_data = pandas.read_csv('CustomerSurveyData.csv',delimiter=',')


#train_df = pandas.crosstab(index = [Customer_survey_data['CreditCard'],Customer_survey_data['JobCategory']],
#                           columns = Customer_survey_data['CarOwnership'], margins = True, dropna = False)

# Check for missing values in the target variable and predictors 
Customer_survey_data['CarOwnership'].isnull().sum()
Customer_survey_data['CreditCard'].isnull().sum()
Customer_survey_data['JobCategory'].isnull().sum()
Customer_survey_data['JobCategory'] = Customer_survey_data['JobCategory'].fillna('Missing')


# Convert the CreditCard and  JobCategory nominal variable into dummy variables
cat_creditCard = Customer_survey_data[['CreditCard']].astype('category')
creditCard_inputs = pandas.get_dummies(cat_creditCard)


cat_jobCategory = Customer_survey_data[['JobCategory']].astype('category')
jobCategory_inputs = pandas.get_dummies(cat_jobCategory)

X_inputs = pandas.concat([creditCard_inputs, jobCategory_inputs], axis=1)
X_inputs

#X_inputs = creditCard_inputs


# Specify the target variable
Y_target = Customer_survey_data['CarOwnership']

# Specify the CART model
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=60616)

# Build the CART model
customerSurveyData_CART = classTree.fit(X_inputs, Y_target)
print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(X_inputs, Y_target)))

# tree diagram
import graphviz
dot_data = tree.export_graphviz(customerSurveyData_CART,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['American Express', 'Discover', 'MasterCard', 'Others', 'Visa','Agriculture', 'Crafts', 'Labor', 'Professional', 'Sales', 'Service','Missing'],
                                class_names = ['Leased', 'None', 'Own'])

graph = graphviz.Source(dot_data)

graph

###############################################################################
# Question 1 - Part (b)
list_cat_creditCard = set(Customer_survey_data['CreditCard'])
#list_cat_creditCard = Customer_survey_data['CreditCard'].unique()
num_cat_creditCard = len(list_cat_creditCard)
print("The categories of credit card: \n", list_cat_creditCard )
print('Number of Categories of creditCard = \n', num_cat_creditCard)
print('Number of  Splits = ', (2**(num_cat_creditCard-1) - 1))
###############################################################################

# Function to calculate entropy
###############################################################################

#crossTable = pandas.crosstab(index = [Customer_survey_data["CreditCard"],Customer_survey_data["JobCategory"]], 
#                             columns = Customer_survey_data['CarOwnership'], margins = True, dropna = False)

def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   dataTable = inData
   dataTable['LE_Split'] = (dataTable.iloc[:,0] == split)

   crossTable = pandas.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   
   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      rowN = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * numpy.log2(proportion)
            rowN += crossTable.iloc[iRow, iColumn]
      print('Row = ', iRow, 'Entropy =', rowEntropy)
      print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
   
   

   return(tableEntropy)      

# Question 1 - Part (c)

creditCard_input = Customer_survey_data[['CreditCard', 'CarOwnership']]

entropy_Table = pandas.DataFrame(columns = ['Entropy'])
#cTable = pandas.DataFrame(list(range(1,16)),columns=['Index'])

cal_entrophy = []
entropy_data = []
account_type = ['American Express', 'Discover', 'MasterCard', 'Others', 'Visa']
subsets = [val for amnt in range(len(account_type)) for val in itertools.combinations(account_type, amnt)]
split = []
for i in range(int(len(subsets)/2 + 1)):
    split.append(list(itertools.chain(subsets[i])) + ['/'] + [sub for sub in account_type if sub not in subsets[i]])
del split[0]
entropy_Table['Splits'] = split

for row in range(15):
    entropy_data.append(entropy_Table['Splits'].str[0:entropy_Table['Splits'][row].index('/')][row])

for i in range(15):
    creditCard_input = creditCard_input.where(creditCard_input != entropy_data[i][0], 'Split')
    if (len(entropy_data[i])==2):
        creditCard_input = creditCard_input.where(creditCard_input != entropy_data[i][1], 'Split')
    cal_entrophy.append(EntropyIntervalSplit(creditCard_input,'Split'))
    creditCard_input = Customer_survey_data[['CreditCard', 'CarOwnership']]
entropy_Table['Entropy'] = cal_entrophy
print(entropy_Table) 

###############################################################################
# Question 1 - Part (D)

ordered_entropy_Table = entropy_Table.sort_values(by = ['Entropy'])
print(ordered_entropy_Table)

###############################################################################
# Question 1 - Part (E)
list_cat_JobCategory  = set(Customer_survey_data['JobCategory'])
num_cat_JobCategory  = len(list_cat_JobCategory )
print("The categories of JobCategory : \n", list_cat_JobCategory  )
print('Number of Categories of JobCategory  = \n', num_cat_JobCategory )
print('Number of  Splits = ', (2**(num_cat_JobCategory -1) - 1))
###############################################################################
# Question 1 - Part (F)
JobCategory_input = Customer_survey_data[['JobCategory', 'CarOwnership']]

entropy_Table_JobCategory = pandas.DataFrame(columns = ['Entropy'])
#cTable = pandas.DataFrame(list(range(1,16)),columns=['Index'])

cal_entrophy_JobCategory= []
entropy_data_JobCategory = []
JobCategory_type = ['Sales', 'Missing', 'Labor', 'Crafts', 'Professional', 'Agriculture', 'Service']
subsets_JobCategory = [val_JobCategory for amnt_JobCategory in range(len(JobCategory_type)) for val_JobCategory in itertools.combinations(JobCategory_type, amnt_JobCategory)]
split = []
for i in range(int(len(subsets_JobCategory)/2 + 1)):
    split.append(list(itertools.chain(subsets_JobCategory[i])) + ['/'] + [sub_JobCategory for sub_JobCategory in JobCategory_type if sub_JobCategory not in subsets_JobCategory[i]])
del split[0]
entropy_Table_JobCategory['Splits'] = split

for row in range(63):
    entropy_data_JobCategory.append(entropy_Table_JobCategory['Splits'].str[0:entropy_Table_JobCategory['Splits'][row].index('/')][row])

for i in range(63):
    JobCategory_input = JobCategory_input.where(JobCategory_input != entropy_data_JobCategory[i][0], 'Split')
    if (len(entropy_data_JobCategory[i])==2):
        JobCategory_input = JobCategory_input.where(JobCategory_input != entropy_data_JobCategory[i][1], 'Split')
    cal_entrophy_JobCategory.append(EntropyIntervalSplit(JobCategory_input,'Split'))
    JobCategory_input = Customer_survey_data[['JobCategory', 'CarOwnership']]
entropy_Table_JobCategory['Entropy'] = cal_entrophy_JobCategory
print(entropy_Table_JobCategory) 

###############################################################################
# Question 1 - Part (G)

ordered_JobCategory_Table = entropy_Table_JobCategory.sort_values(by = ['Entropy'])
print(ordered_JobCategory_Table)