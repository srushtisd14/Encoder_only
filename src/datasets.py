DATASET_CONFIG = {
    "sst2": {
        "name": "glue",
        "subset": "sst2",
        "text1": "sentence",
        "text2": None
    },

    "mrpc": {
        "name": "glue",
        "subset": "mrpc",
        "text1": "sentence1",
        "text2": "sentence2"
    },

    "qnli": {
        "name": "glue",
        "subset": "qnli",
        "text1": "question",
        "text2": "sentence"
    },

    "mnli": {
        "name": "glue",
        "subset": "mnli",
        "text1": "premise",
        "text2": "hypothesis"
    },

    "imdb": {
        "name": "stanfordnlp/imdb",
        "subset": None,
        "text1": "text",
        "text2": None
    },

    "ag_news": {
        "name": "fancyzhx/ag_news",
        "subset": None,
        "text1": "text",
        "text2": None
    }
}