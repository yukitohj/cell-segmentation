
def read_csvs(csv_paths: list, split=False, shuffle=False, seed=0):
    import pandas as pd
    from hydra.utils import to_absolute_path

    csvs = [pd.read_csv(to_absolute_path(csv_path)) for csv_path in csv_paths]
    df = pd.concat(csvs, axis=0).reset_index(drop=True)

    if shuffle:
        df = df.sample(frac=1, ignore_index=True, random_state=seed)

    if split:
        t = int(len(df)*0.7)
        train = (df[:t]['input'].reset_index(drop=True), df[:t]['label'].reset_index(drop=True))
        test = (df[t:]['input'].reset_index(drop=True), df[t:]['label'].reset_index(drop=True))
        return train, test

    return df['input'].reset_index(drop=True), df['label'].reset_index(drop=True)
