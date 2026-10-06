# 11_Association_Rule_Mining.py
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

transactions = [
    ['Bread', 'Milk'],
    ['Bread', 'Butter', 'Juice', 'Eggs'],
    ['Milk', 'Butter', 'Juice', 'Coke'],
    ['Bread', 'Milk', 'Butter', 'Juice'],
    ['Bread', 'Milk', 'Butter', 'Coke']
]
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)
print('Transaction Data:')
print(df)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print('\nFrequent Itemsets:')
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
print('\nAssociation Rules:')
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
