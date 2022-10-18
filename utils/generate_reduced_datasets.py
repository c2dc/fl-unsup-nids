import pandas as pd

botiot = pd.read_csv("../sampled_datasets/botiot_sampled.csv.gz")
cicids = pd.read_csv("../sampled_datasets/cicids_sampled.csv.gz")
nb15 = pd.read_csv("../sampled_datasets/nb15_sampled.csv.gz")
toniot = pd.read_csv("../sampled_datasets/toniot_sampled.csv.gz")

toniot_dist = [
    ["xss", 48652],
    ["Benign", 54024],
    ["scanning", 21335],
    ["DDoS", 40000],
    ["DoS", 12665],
    ["injection", 13686],
    ["password", 22557],
    ["Backdoor", 352],
    ["mitm", 124],
    ["ransomware", 65]
]

botiot_dist = [
    ["DDoS", 84552],
    ["DoS", 167004],
    ["Reconnaissance", 25907],
    ["Benign", 1285],
    ["Theft", 33]
]

nb15_dist = [
    ["Benign", 455751],
    ["Exploits", 5902],
    ["Reconnaissance", 1309],
    ["Fuzzers", 3039],
    ["Backdoor", 65],
    ["Analysis", 71],
    ["Generic", 576],
    ["Shellcode", 127],
    ["DoS", 661],
    ["Worms", 38]
]

cicids_dist = [
    ["Benign", 154649],
    ["DDoS", 13828],
    ["Infilteration", 1087],
    ["DoS", 4855],
    ["Brute Force", 1202],
    ["Bot", 1510],
    ["injection", 8]
]


print("> TON-IoT")
for attack in toniot_dist:
    num_samples_attack = toniot[toniot['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling TON-IoT ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    toniot = toniot.drop(toniot[toniot['Attack'] == attack[0]].sample(n=q).index)
toniot.to_csv("../reduced_datasets/toniot_bench_dnn.csv.gz", index=False)

print("> Bot-IoT")
for attack in botiot_dist:
    num_samples_attack = botiot[botiot['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling Bot-IoT ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    botiot = botiot.drop(botiot[botiot['Attack'] == attack[0]].sample(n=q).index)
botiot.to_csv("../reduced_datasets/botiot_bench_dnn.csv.gz", index=False)

print("> UNSW-NB15")
for attack in nb15_dist:
    num_samples_attack = nb15[nb15['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling UNSW-NB15 ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    nb15 = nb15.drop(nb15[nb15['Attack'] == attack[0]].sample(n=q).index)
nb15.to_csv("../reduced_datasets/nb15_bench_dnn.csv.gz", index=False)

print("> CIC-IDS-2018")
for attack in cicids_dist:
    num_samples_attack = cicids[cicids['Attack'] == attack[0]].shape[0]
    q = num_samples_attack - attack[1]
    print(">> Sampling CIC-IDS-2018 ({}) - Num of samples {} | Required Samples {} | To be removed {}".format(attack[0], num_samples_attack, attack[1], q))
    cicids = cicids.drop(cicids[cicids['Attack'] == attack[0]].sample(n=q).index)
cicids.to_csv("../reduced_datasets/cicids_bench_dnn.csv.gz", index=False)
