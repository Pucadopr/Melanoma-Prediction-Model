def seed_all(seed: int = 1992):

    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(
        seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, model, config: type):
        self.model = model
        # self.device = device
        self.config = config
        self.epoch = 0
        self.best_acc = 0
        self.best_loss = 10**5

        # TODO consider moving these to config class
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=config.lr,
                                           weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.8,
            patience=1,
            verbose=True,
            min_lr=1e-8)

        self.log("Trainer prepared. We are using {} device.".format(
            self.config.device))

    def fit(self, train_loader, val_loader, fold: int):

        self.log("Training on Fold {}".format(fold + 1))

        for epoch in range(self.config.n_epochs):
            # Getting the learning rate after each epoch!
            lr = self.optimizer.param_groups[0]["lr"]
            timestamp = datetime.fromtimestamp(time.time())
            # printing the lr and the timestamp after each epoch.
            self.log("\n{}\nLR: {}".format(timestamp, lr))
            # start time of training on the training set
            train_start_time = time.time()

            # train one epoch on the training set
            avg_train_loss, avg_train_acc_score = self.train_one_epoch(
                train_loader)
            # end time of training on the training set
            train_end_time = time.time()

            # formatting time to make it nicer
            train_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(train_end_time - train_start_time))
            self.log(
                "[RESULT]: Train. Epoch {} | Avg Train Summary Loss: {:.6f} | Train Accuracy: {:6f} | Time Elapsed: {}"
                .format(self.epoch + 1, avg_train_loss, avg_train_acc_score,
                        train_elapsed_time))

            val_start_time = time.time()
            # note here has val predictions... in actual fact it is repeated because its same as avg_val_acc_score
            avg_val_loss, avg_val_acc_score, val_predictions = self.valid_one_epoch(
                val_loader)
            # not sure if it is good practice to write it here
            self.val_predictions = val_predictions
            val_end_time = time.time()
            val_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(val_end_time - val_start_time))

            self.log(
                "[RESULT]: Validation. Epoch: {} | Avg Validation Summary Loss: {:.6f} | Validation Accuracy: {:.6f} | Time Elapsed: {}"
                .format(self.epoch + 1, avg_val_loss, avg_val_acc_score,
                        val_elapsed_time))

            # note here we use avg_val_loss, not train_val_loss! It is just right to use val_loss as benchmark
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                # decided to remove epoch here as epoch can be saved in the model later in self.save
                # also this will overwrite everytime there is a better weight.
                # TODO consider including epoch number inside, and call this epoch number as well
                # through self.load to load the weights in curr_fold_best_checkpoint
                self.save("{}_best_loss_fold_{}.pt".format(
                    self.config.effnet, fold))

            if self.best_acc < avg_val_acc_score:
                self.best_acc = avg_val_acc_score
                # TODO consider saving these weights as well.

            # this part not so clear yet, figure this out on why .step(loss) vs .step() in train epoch
            if self.config.val_step_scheduler:
                self.scheduler.step(avg_val_loss)

            # end of training, epoch + 1 so that self.epoch can be updated.
            self.epoch += 1

        # this is where we end the epoch training for the current fold/model, therefore
        # we can call the final "best weight saved" by this exact name that we saved earlier on.
        curr_fold_best_checkpoint = self.load("{}_best_loss_fold_{}.pt".format(
            self.config.effnet, fold))
        # return the checkpoint for further usage.
        return curr_fold_best_checkpoint

    def train_one_epoch(self, train_loader):

        # set to train mode
        self.model.train()

        # log metrics
        summary_loss = AverageLossMeter()
        accuracy_scores = AccuracyMeter()

        # timer
        start_time = time.time()

        # looping through train loader for one epoch, steps is the number of times to go through each epoch
        for step, (image_ids, images, labels) in enumerate(train_loader):

            
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            
            batch_size = images.shape[0]


            logits = self.model(images)

            
            loss = self.criterion(input=logits, target=labels)
            summary_loss.update(loss.item(), batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            
            y_true = labels.cpu().numpy()
            softmax_preds = torch.nn.Softmax(dim=1)(input=logits).to("cpu").detach().numpy()
            y_preds = softmax_preds.argmax(1)
            
            accuracy_scores.update(y_true, y_preds, batch_size=batch_size)
            

            # not too sure yet KIV
            if self.config.train_step_scheduler:
                self.scheduler.step()

            # measure elapsed time
            end_time = time.time()

            if config.verbose:
                if (step % config.verbose_step) == 0:
                    print(
                        f"Train Steps {step}/{len(train_loader)}, " +
                        f"summary_loss: {summary_loss.avg:.3f}, acc: {accuracy_scores.avg:.3f} "
                        + f"time: {(end_time - start_time):.3f}",
                        end="\r",
                    )

        return summary_loss.avg, accuracy_scores.avg

    def valid_one_epoch(self, val_loader):

        # set to eval mode
        self.model.eval()

        # log metrics
        summary_loss = AverageLossMeter()
        accuracy_scores = AccuracyMeter()

        # timer
        start_time = time.time()
        # predictions list to append for oof later
        val_preds_list = []

        # off gradients for torch when validating
        with torch.no_grad():
            for step, (image_ids, images, labels) in enumerate(val_loader):

                images = images.to(self.config.device)
                labels = labels.to(self.config.device)
                batch_size = images.shape[0]

                logits = self.model(images)
                loss = self.criterion(input=logits, target=labels)
                summary_loss.update(loss.item(), batch_size)

                y_true = labels.cpu().numpy()
                # Write that we do not need to detach here as no gradients involved.
                # Basically torch.nn.Softmax(dim=1)(input=logits).to("cpu").detach.numpy()
                softmax_preds = torch.nn.Softmax(dim=1)(
                    input=logits).to("cpu").numpy()
                y_preds = softmax_preds.argmax(1)
                accuracy_scores.update(y_true, y_preds, batch_size=batch_size)

                val_preds_list.append(softmax_preds)

                end_time = time.time()

                if config.verbose:
                    if (step % config.verbose_step) == 0:
                        print(
                            f"Validation Steps {step}/{len(val_loader)}, " +
                            f"summary_loss: {summary_loss.avg:.3f}, val_acc: {accuracy_scores.avg:.6f} "
                            + f"time: {(end_time - start_time):.3f}",
                            end="\r",
                        )

            val_predictions = np.concatenate(val_preds_list)
           
        return summary_loss.avg, accuracy_scores.avg, val_predictions

    def save_model(self, path):
        self.model.eval()
        torch.save(self.model.state_dict(), path)

    # will save the weight for the best val loss and corresponding oof preds
    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_acc": self.best_acc,
                "best_loss": self.best_loss,
                "epoch": self.epoch,
                "oof_preds": self.val_predictions,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        return checkpoint


    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.config.log_path, "a+") as logger:
            logger.write(f"{message}\n")


def train_on_fold(config, fold: int):
    model = CustomEfficientNet(config=config, pretrained=True)
    # consider remove if clause?
    if torch.cuda.is_available():
        model.cuda()

    transforms_train, transforms_val = get_transforms(config)

    train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
    val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)

    train_set = Melanoma(train_df, config, transforms=transforms_train, test=False, albu_norm=False)
    train_loader = DataLoader(train_set,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=4,
                              worker_init_fn=seed_worker)

    val_set = Melanoma(val_df, config, transforms=transforms_val, test=False, albu_norm=False)
    val_loader = DataLoader(val_set,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4,
                            worker_init_fn=seed_worker)


    cassava_trainer = Trainer(model=model, config=config)

    curr_fold_best_checkpoint = cassava_trainer.fit(train_loader, val_loader,
                                                    fold)

    # loading checkpoint for all 10 epochs for this current fold

    val_df[[str(c) for c in range(config.num_classes)
            ]] = curr_fold_best_checkpoint["oof_preds"]
    val_df["preds"] = curr_fold_best_checkpoint["oof_preds"].argmax(1)

    return val_df


def get_acc_score(y_true, y_pred):
    return sklearn.metrics.accuracy_score(y_true, y_pred)


def get_result(result_df):
    preds = result_df["preds"].values
    labels = result_df[config.class_col_name].values
    score = get_acc_score(labels, preds)
    return score


def train_loop(df_folds, config, fold_num: int = None, train_one_fold=False):
    # here The CV score is the average of the validation fold metric.
    cv_score_list = []
    oof_df = pd.DataFrame()
    if train_one_fold:
        _oof_df = train_on_fold(fold_num)
        curr_fold_best_score = get_result(_oof_df)
        print("Fold {} OOF Score is {}".format(fold_num + 1,
                                               curr_fold_best_score))
    else:
        #for fold in sorted(df_folds["fold"].unique()):
        for fold in range(config.num_folds):
            # note very carefully you need to add 1 here. because df_folds is 1,2,3,4,5
            _oof_df = train_on_fold(config, fold)
            #_oof_df = train_on_fold(config, fold+1)
            oof_df = pd.concat([oof_df, _oof_df])
            curr_fold_best_score = get_result(_oof_df)
            cv_score_list.append(curr_fold_best_score)
            print("\n\n\nOOF Score for Fold {}: {}\n\n\n".format(
                fold + 1, curr_fold_best_score))

    print("CV score", np.mean(cv_score_list))
    print("Variance", np.var(cv_score_list))
    print("Five Folds OOF", get_result(oof_df))
    oof_df.to_csv("oof.csv")
