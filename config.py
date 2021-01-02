import torch

class GlobalConfig:
    seed = 1958
    num_classes = 2
    batch_size = 16
    n_epochs = 5
    lr = 0.00003
    scheduler = "CosineAnnealingWarmRestarts"
    train_step_scheduler = False
    val_step_scheduler = True
    T_0 = 10 
    min_lr = 1e-6
    weight_decay = 1e-6
    image_size = 512
    resize = 256
    crop_size = {128: 110, 256: 200, 512: 400}
    verbose = 1
    verbose_step = 1
    num_folds = 5
    class_col_name = "target"
    log_path = "./log.txt"
    train_path = '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma'

    csv_path = "../input/melanoma-external-malignant-256/train_concat.csv/"
    save_path = "./"
    # test_path = '../input/cassava-leaf-disease-classification/test_images/'
    effnet = "tf_efficientnet_b2_ns"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weight_path = '../input/efficientnet-weights/tf_efficientnet_b2_ns-00306e48.pth'