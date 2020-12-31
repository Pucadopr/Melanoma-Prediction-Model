if __name__ == "__main__":
    config = GlobalConfig
    seed_all(seed=config.seed)
    train_csv = pd.read_csv('../input/melanoma-merged-external-data-512x512-jpeg/folds.csv')
    df_folds = train_csv.copy()
    df_folds = df_folds.rename(columns={'image_id': 'image_name'})
#     skf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
#     for fold, (train_index, val_index) in enumerate(skf.split(df_folds)):
#         df_folds.loc[val_index, 'fold'] = int(fold)
#     df_folds['fold'] = df_folds['fold'].astype(int)
    print(df_folds.groupby(['fold', config.class_col_name]).size())
    train_csv.target.value_counts()
    train_single_fold = train_on_fold(config, fold=1)
    #train_all_folds = train_loop(df_folds,config)