import pandas as pd
from fast_ml.model_development import train_valid_test_split


def write_all(train_path, test_path, valid_path, all_path):
    with open(all_path, "a") as all_file:
        train_content = []
        train_file = open(train_path, "r")
        for line in train_file:
            l = line.strip().split("\t")
            s = l[0]
            r = l[1]
            if (s, r) not in train_content:
                train_content.append((s, r))
            all_file.write(line)
        train_file.close()
        test_content = []
        test_file = open(test_path, "r")
        for line in test_file:
            l = line.strip().split("\t")
            s = l[0]
            r = l[1]
            if (s, r) not in test_content:
                test_content.append((s, r))
            if (s, r) not in train_content:
                all_file.write(line)
        test_file.close()
        valid_file = open(valid_path, "r")
        for line in valid_file:
            l = line.strip().split("\t")
            s = l[0]
            r = l[1]
            if (s, r) not in test_content and (s, r) not in train_content:
                all_file.write(line)
        valid_file.close()
        

def make_permutation(all_path, train_path, test_path, valid_path):
    df = pd.read_table(all_path, header=0, dtype=str, keep_default_na=False, on_bad_lines="warn")
    x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(df, target="s", train_size=0.8, valid_size=0.1, test_size=0.1)
    # print("Training:")
    # print(f"X: {len(x_train.index)} Y: {len(y_train.index)}")
    # print("Validation:")
    # print(f"X: {len(x_valid.index)} Y: {len(y_valid.index)}")
    # print("Testing:")
    # print(f"X: {len(x_test.index)} Y: {len(y_test.index)}")
    x_train.insert(loc=0, column="s", value=y_train)
    x_valid.insert(loc=0, column="s", value=y_valid)
    x_test.insert(loc=0, column="s", value=y_test)
    x_train.to_csv(train_path, sep="\t", header=None, index=False)
    x_valid.to_csv(valid_path, sep="\t", header=None, index=False)
    x_test.to_csv(test_path, sep="\t", header=None, index=False)


if __name__ == "__main__":
    write_all(train_path=r"data\raw\train",
              test_path=r"data\raw\test",
              valid_path=r"data\raw\valid",
              all_path=r"data\raw\all")
    make_permutation(all_path=r"data\raw\all", 
                     train_path=r"data\raw\train",
                     test_path=r"data\raw\test",
                     valid_path=r"data\raw\valid")
