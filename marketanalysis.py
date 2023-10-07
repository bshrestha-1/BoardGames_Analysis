

# Convert transactions to the format needed for MBA
def get_transaction_data(data_frame):
    transactions = data_frame['transactions'].tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

train_transactions = get_transaction_data(train_df)
test_transactions = get_transaction_data(test_df)

# Generating rules from the training set
frequent_itemsets_train = apriori(train_transactions, min_support=0.01, use_colnames=True)
rules_train = association_rules(frequent_itemsets_train, metric="confidence", min_threshold=0.5)

# Evaluating the rules on the test set
# First, we need to adjust how we evaluate the presence of antecedents:
def match_antecedents(row, rules):
    for antecedents in rules['antecedents']:
        if row[list(antecedents)].all():
            return True
    return False

test_transactions['matches'] = test_transactions.apply(match_antecedents, axis=1, rules=rules_train)
test_support = test_transactions['matches'].mean()

print("Test Support:", test_support)
print(rules_train[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

rules_train.to_csv('association_rules.csv', index=False)

