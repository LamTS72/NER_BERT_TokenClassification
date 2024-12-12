class ConfigModel():
    NAME_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    MODEL_NAME = "bert-base-cased"
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    METRICs = "seqeval"
    PATH_TENSORBOARD = "data_run"
    PATH_SAVE = "runs/token_classifier"
    NUM_WARMUP_STEPS = 0

class ConfigHelper():
    TOKEN_HF = "xxx"
    AUTHOR = "Chessmen"