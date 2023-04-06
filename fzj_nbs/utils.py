from pathlib import Path

def get_full_history(hist_dir, verbose=False):
    jsons = list(hist_dir.glob("history*.json"))
    if verbose:
        print(f"{hist_dir.parent} has {len(jsons)} hisotries")
    jsons.sort(key=lambda x: int(x.name.split("_")[1].split(".")[0]))  # sort according to epoch number

    # initialize a dict with correct keys and empty lists as values
    with open(jsons[0]) as h:
        keys = json.load(h).keys()
    full_history = {key: [] for key in keys}

    # join epoch values to a full history
    for path in jsons:
        with open(path) as h:
            epoch = json.load(h)
            for key in epoch.keys():
                full_history[key].append(epoch[key])

    reg_loss = np.sum(
        np.array([full_history["{}_loss".format(l)] for l in ["energy", "pt", "eta", "sin_phi", "cos_phi", "charge"]]),
        axis=0,
    )
    val_reg_loss = np.sum(
        np.array(
            [full_history["val_{}_loss".format(l)] for l in ["energy", "pt", "eta", "sin_phi", "cos_phi", "charge"]]
        ),
        axis=0,
    )
    full_history.update({"reg_loss": reg_loss})
    full_history.update({"val_reg_loss": val_reg_loss})

    return full_history, len(jsons)


def get_histories(train_dirs):
    train_dirs = [Path(train_dir) for train_dir in train_dirs]
    histories = []

    for train_dir in train_dirs:
        hist, N = get_full_history(hist_dir=train_dir / "history")
        histories.append(hist)

    return histories


def count_skipped_configurations(exp_dir):
    skiplog_file_path = Path(exp_dir) / "skipped_configurations.txt"
    if skiplog_file_path.exists():
        with open(skiplog_file_path, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                if line == "#"*80 + "\n":
                    count += 1
        if count % 2 != 0:
            print("WARNING: counts is not divisible by two")
        print("Number of skipped configurations: {}".format(count // 2))
    else:
        print("Could not find {}".format(str(skiplog_file_path)))
