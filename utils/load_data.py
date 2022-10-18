import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

seed=42

not_applicable_features = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack', 'Dataset', 'Label']

def remove_features(df, full, feats=not_applicable_features):
    if full:
        feats.remove('Dataset')
    
    X = df.drop(columns=feats)
    y = df['Label']
    return X, y

def train_test_scaled(X, y, test_size):
    scaler = MinMaxScaler()
    indices = list(X.index)
    X_train, X_test, y_train, y_test, _, test_index = train_test_split(X, y, indices, test_size=test_size, 
                                                       random_state=seed, stratify=y)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, test_index


def load_data(cid, info=True, test_size=0.2, full=False):
    if ("NF-BoT-IoT-v2.csv.gz" in cid):
        df = pd.read_csv(cid, low_memory=True, nrows=18893708) # Same nrows as CSE-CIC-IDS-2018
    else:
        df = pd.read_csv(cid, low_memory=True)

    df.dropna(inplace=True)
    if info:
        print(">> \033[1m {} \033[0m - Total samples: {}, Benign: {}, Malicious: {}, Labels: {}"\
            .format(cid[cid.rfind("/")+1:cid.find(".csv")], df.shape[0], sum(df.Label == 0), \
            sum(df.Label == 1), sorted(list(df.Attack.unique().astype(str)))))
    X, y = remove_features(df, full=full)
    x_train, y_train, x_test, y_test, test_index = train_test_scaled(X, y, test_size)


    if info:
        ref = cid[cid.rfind("/") + 1:cid.find(".csv")]
        df['Attack'].iloc[test_index].to_csv("./error_analysis/" + ref + "_test_classes.csv")

    return x_train, y_train, x_test, y_test
