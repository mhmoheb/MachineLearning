import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

Groceries_df = pd.read_csv('Groceries.csv', delimiter=',')
num_customer = Groceries_df.groupby(['Customer'])['Item'].nunique()
print(num_customer)
#Q2-A
print("\n Answer to question 2 part A :\n")
frq_cnt = pd.value_counts(num_customer).reset_index()
frq_cnt.columns = ['Item', 'Frequency']
frq_cnt = frq_cnt.sort_values(by = ['Item'])
print(frq_cnt)
print(num_customer.describe())
###############################################################################
#Q2-B
print("\n Answer to question 2 part B :\n")
print("Unique Items = ", len(set(Groceries_df['Item'])))
###############################################################################
#Q2-C
print("\n Answer to question 2 part C :\n")
plt.hist(x=num_customer)
plt.grid(True)
plt.show()

###############################################################################
ListItem = Groceries_df.groupby(['Customer'])['Item'].apply(list).values.tolist()
tran_encoder = TransactionEncoder()
tran_encoder_list = tran_encoder.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(tran_encoder_list, columns=tran_encoder.columns_)


# Q2-D)
print("\n Answer to question 2 part D :\n")
frq_itemsets = apriori(ItemIndicator, min_support = 75/len(num_customer), use_colnames = True)
frq_itemsets['length'] = frq_itemsets['itemsets'].apply(lambda x: len(x))
#number of itemdets
print("The number of itemsets: " )
print(len(frq_itemsets))
#higest k value
print("The higest K value : " )
print( max(frq_itemsets['length']))

###############################################################################
#Q2-E)
print("\n Answer to question 2 part E :\n")

assoc_rules = association_rules(frq_itemsets, metric = "confidence", min_threshold = 0.01)
assoc_rules["antecedent_len"] = assoc_rules["antecedents"].apply(lambda x: len(x))
assoc_rules["consequent_len"] = assoc_rules["consequents"].apply(lambda x: len(x))
assoc_rules[ (assoc_rules['antecedent_len'] >= 1) & (assoc_rules['consequent_len'] >= 1) ]
print(assoc_rules.shape[0])
###############################################################################
#Q2-F)
print("\n Answer to question 2 part F :\n")
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

###############################################################################
#Q2-G)
print("\n Answer to question 2 part G :\n")
assoc_rules = association_rules(frq_itemsets, metric = "confidence", min_threshold = 0.6)
print(assoc_rules)

