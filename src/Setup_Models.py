from unsloth import FastLanguageModel

_ , _ = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

_ , _ = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3.5-mini-instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

_ , _ = FastLanguageModel.from_pretrained(
    model_name = "OpenLLM-Ro/RoMistral-7b-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

_ , _ = FastLanguageModel.from_pretrained(
    model_name = "OpenLLM-Ro/RoLlama3.1-8b-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)
