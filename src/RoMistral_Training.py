import pandas as pd
import numpy as np
import torch
import gc
import time
from sklearn.model_selection import KFold
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from google.colab import drive
from tqdm import tqdm
import warnings
import os
from transformers import logging as hf_logging

drive.mount('/content/drive')
CALE_SALVARE_DRIVE = "/content/drive/MyDrive/Licenta/RoMistral"

MODEL_NAME = "OpenLLM-Ro/RoMistral-7b-Instruct"
DATASET_PATH = "/content/drive/MyDrive/Licenta/dataset_ro_3coloane_800.csv"
LIMBA_EVALUARE = "ro"
MAX_SEQ_LENGTH = 2048

LEARNING_RATES = [2e-5, 5e-5]
WEIGHT_DECAYS = [0.01, 0.05]
K_FOLDS = 5
MAX_STEPS = 80

df = pd.read_csv(DATASET_PATH)
dataset_complet = Dataset.from_pandas(df)
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

rezultate_finale = []
cel_mai_bun_scor_global = 0.0

for lr in LEARNING_RATES:
    for wd in WEIGHT_DECAYS:
        print(f"\n GRID SEARCH: LR = {lr} | WD = {wd} ---")

        scoruri_rouge_grid = []
        scoruri_bert_grid = []
        timpi_antrenament = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_complet)):
            print(f"\n Rulare Fold {fold + 1}/{K_FOLDS}")

            train_subset = dataset_complet.select(train_idx)
            test_subset_complet = dataset_complet.select(test_idx)

            NUMAR_EVALUARI = 30
            test_subset = test_subset_complet.shuffle(seed=42).select(range(NUMAR_EVALUARI))


            def formator_unsloth(exemple):
                texte_native = []
                for inst, doc, rez in zip(exemple['instructiune'], exemple['document'], exemple['rezumat']):

                    text_lipit = f"{inst}\n\nDOCUMENT:\n{doc}"

                    conversatie = [
                        {"role": "user", "content": text_lipit},
                        {"role": "assistant", "content": rez}
                    ]
                    text_nativ = tokenizer.apply_chat_template(conversatie, tokenize=False, add_generation_prompt=False)
                    texte_native.append(text_nativ)
                return texte_native

            print("Incarcare model baseline")
            model_curat, tokenizer = FastLanguageModel.from_pretrained(
                model_name = MODEL_NAME,
                max_seq_length = MAX_SEQ_LENGTH,
                dtype = None,
                load_in_4bit = True,
            )

            train_dataset = train_subset.map(
                lambda x: {"text": formator_unsloth(x)},
                batched = True
            )

            def functie_tranzitie(exemple):
                return exemple["text"]

            model = FastLanguageModel.get_peft_model(
                model_curat,
                r = 32,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha = 32,
                lora_dropout = 0,
                bias = "none",
                use_gradient_checkpointing = "unsloth",
                random_state = 3407,
            )

            trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = train_dataset,
                dataset_text_field = "text",
                formatting_func = functie_tranzitie,
                max_seq_length = MAX_SEQ_LENGTH,
                dataset_num_proc = 2,
                args = TrainingArguments(
                    per_device_train_batch_size = 2,
                    gradient_accumulation_steps = 4,
                    warmup_steps = 5,
                    max_steps = MAX_STEPS,
                    learning_rate = lr,
                    weight_decay = wd,
                    fp16 = not torch.cuda.is_bf16_supported(),
                    bf16 = torch.cuda.is_bf16_supported(),
                    logging_steps = 20,
                    optim = "adamw_8bit",
                    output_dir = "temp_outputs",
                    save_strategy = "no",
                    remove_unused_columns = False,
                    seed = 3407,
                ),
            )

            start_time = time.time()
            trainer.train()
            timpi_antrenament.append(time.time() - start_time)

            FastLanguageModel.for_inference(model)

            scoruri_rouge_fold_curent = []
            referinte_fold = []
            generate_fold = []

            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)

            for i in tqdm(range(len(test_subset)), desc="Generare Rezumate"):
                inst = test_subset[i]['instructiune']
                doc = test_subset[i]['document']
                referinta = test_subset[i]['rezumat']

                text_lipit = f"{inst}\n\nDOCUMENT:\n{doc}"
                conversatie_test = [{"role": "user", "content": text_lipit}]

                prompt_test_nativ = tokenizer.apply_chat_template(conversatie_test, tokenize=False, add_generation_prompt=True)

                inputs_std = tokenizer(prompt_test_nativ, return_tensors="pt").to("cuda")
                outputs_std = model.generate(**inputs_std, max_new_tokens=150, use_cache=True, pad_token_id=tokenizer.eos_token_id)

                lungime_input = inputs_std['input_ids'].shape[1]
                generat_std = tokenizer.decode(outputs_std[0][lungime_input:], skip_special_tokens=True).strip()

                valoare_rouge = scorer.score(referinta, generat_std)['rouge1'].fmeasure
                scoruri_rouge_fold_curent.append(valoare_rouge)

                referinte_fold.append(referinta)
                generate_fold.append(generat_std)

            print("Calculare BERTScore")
            _, _, F1_bert = bert_score_func(generate_fold, referinte_fold, lang=LIMBA_EVALUARE, verbose=False)
            scoruri_bert_fold_curent = F1_bert.tolist()

            medie_rouge_fold = np.mean(scoruri_rouge_fold_curent)
            medie_bert_fold = np.mean(scoruri_bert_fold_curent)
            scor_mediu_fold = (medie_rouge_fold + medie_bert_fold) / 2

            print(f"Performanta Fold {fold + 1} | ROUGE-1: {medie_rouge_fold:.4f} | BERTScore: {medie_bert_fold:.4f} | MEDIE: {scor_mediu_fold:.4f}")

            scoruri_rouge_grid.append(medie_rouge_fold)
            scoruri_bert_grid.append(medie_bert_fold)

            if scor_mediu_fold > cel_mai_bun_scor_global:
                print(f"NOU RECORD GASIT IN FOLD {fold+1} (Scor: {scor_mediu_fold:.4f})")
                cel_mai_bun_scor_global = scor_mediu_fold
                model.save_pretrained(CALE_SALVARE_DRIVE)
                tokenizer.save_pretrained(CALE_SALVARE_DRIVE)

            del trainer
            del model
            del model_curat
            if 'inputs_std' in locals(): del inputs_std
            if 'outputs_std' in locals(): del outputs_std
            if 'F1_bert' in locals(): del F1_bert
            gc.collect()
            torch.cuda.empty_cache()

        medie_rouge_final = np.mean(scoruri_rouge_grid)
        medie_bert_final = np.mean(scoruri_bert_grid)

        rezultate_finale.append({
            'LR': lr, 'WD': wd,
            'ROUGE-1 F1': f"{medie_rouge_final:.4f} ± {np.std(scoruri_rouge_grid):.4f}",
            'BERTScore F1': f"{medie_bert_final:.4f} ± {np.std(scoruri_bert_grid):.4f}",
            'Scor Mediu': f"{(medie_rouge_final + medie_bert_final) / 2:.4f}",
            'Timp Mediu (s)': f"{np.mean(timpi_antrenament):.2f}"
        })

print("\n=== REZULTATE FINALE GRID SEARCH ===")
print(pd.DataFrame(rezultate_finale).to_string(index=False))
print(f"Cel mai bun model a fost salvat la: {CALE_SALVARE_DRIVE}")