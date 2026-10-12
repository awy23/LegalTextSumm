import gc
import torch
import gradio as gr
from unsloth import FastLanguageModel
from deep_translator import GoogleTranslator
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from google.colab import drive

drive.mount('/content/drive')

current_model = None
current_tokenizer = None
loaded_model_name = None

MODEL_PATHS = {
    "RoMistral-7b (RO)": {
        "path": "/content/drive/MyDrive/Licenta/RoMistral", 
        "lang": "ro"
    },
    "RoLLaMA-3.1-8b (RO)": {
        "path": "/content/drive/MyDrive/Licenta/RoLlama3.1", 
        "lang": "ro"
    },
    "Phi-3.5-Instruct (EN)": {
        "path": "/content/drive/MyDrive/Licenta/Phi3.5", 
        "lang": "en"
    },
    "LLaMA-3.1-8b (EN)": {
        "path": "/content/drive/MyDrive/Licenta/Llama3.1", 
        "lang": "en"
    }
}

PROMPT_RO = """Ești un expert juridic. Rezumă punctele principale ale următorului document.
Regula 1: Nu folosi fraze introductive.
Regula 2: Păstrează textul concis.
Regula 3: Asigură-te că toate numerele, termenele limită și drepturile legale sunt păstrate exact și sunt incluse explicit în rezumatul tău.

Document:
{text_document}"""

PROMPT_EN = """As a legal expert, summarize the main points of the following document.
Rule 1: Do not use introductory phrases.
Rule 2: Keep the text concise.
Rule 3: Ensure all numbers, deadlines, and legal rights are preserved exactly and are explicitly included in your summary.

Document:
{text_document}"""

def free_vram():
    global current_model, current_tokenizer, loaded_model_name
    if current_model is not None:
        del current_model
        del current_tokenizer
        current_model = None
        current_tokenizer = None
        loaded_model_name = None
        gc.collect()
        torch.cuda.empty_cache()

def load_model(selected_model_name):
    global current_model, current_tokenizer, loaded_model_name

    if loaded_model_name == selected_model_name:
        return current_model, current_tokenizer

    free_vram()
    model_config = MODEL_PATHS[selected_model_name]

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config["path"],
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        raise RuntimeError(f"Eroare la incarcarea modelului: {e}")

    FastLanguageModel.for_inference(model)
    model.config.use_cache = True

    current_model = model
    current_tokenizer = tokenizer
    loaded_model_name = selected_model_name

    return current_model, current_tokenizer

def generate_core_summary(model, tokenizer, text, prompt_template, active_model_name):
    if "{text_document}" in prompt_template:
        instruction = prompt_template.split("{text_document}")[0].strip()
        for suffix in ["Document:", "DOCUMENT:", "document:"]:
            if instruction.endswith(suffix):
                instruction = instruction[:-len(suffix)].strip()
                break
    else:
        instruction = prompt_template.strip()

    if "RoMistral" in active_model_name or "Mistral" in active_model_name:
        combined_text = f"{instruction}\n\nDOCUMENT:\n{text.strip()}"
        chat = [{"role": "user", "content": combined_text}]
    else:
        chat = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text.strip()}
        ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

    if "RoMistral" in active_model_name:
        gen_args = {
            "max_new_tokens": 150,
            "do_sample": False,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 20,
            "repetition_penalty": 1.15,
        }
    elif "RoLLaMA" in active_model_name:
        gen_args = {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 20,
            "repetition_penalty": 1.15,
        }
    else:
        gen_args = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
        }

    gen_args["use_cache"] = True
    gen_args["pad_token_id"] = tokenizer.eos_token_id
    gen_args["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_args)

    summary_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(summary_tokens, skip_special_tokens=True).strip()

def calculate_metrics(generated_text, reference_text, eval_language="ro"):
    if not reference_text or len(reference_text.strip()) == 0:
        return "Nu a fost introdus un text de referinta. Metricile nu pot fi calculate."

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(reference_text, generated_text)

    P, R, F1 = bert_score_func([generated_text], [reference_text], lang=eval_language, verbose=False)

    result = (
        f"--- Scoruri ROUGE ---\n"
        f"ROUGE-1: Precision={scores['rouge1'].precision:.4f}, Recall={scores['rouge1'].recall:.4f}, F1={scores['rouge1'].fmeasure:.4f}\n"
        f"ROUGE-2: Precision={scores['rouge2'].precision:.4f}, Recall={scores['rouge2'].recall:.4f}, F1={scores['rouge2'].fmeasure:.4f}\n"
        f"ROUGE-L: Precision={scores['rougeL'].precision:.4f}, Recall={scores['rougeL'].recall:.4f}, F1={scores['rougeL'].fmeasure:.4f}\n\n"
        f"--- Scoruri BERTScore ---\n"
        f"Precision: {P.item():.4f}\n"
        f"Recall: {R.item():.4f}\n"
        f"F1-Score: {F1.item():.4f}"
    )
    return result

def process_request(document_text, reference_text, model_option):
    if not document_text or len(document_text.strip()) == 0:
        return "Eroare: Lipsa text sursa.", ""

    final_summary = ""
    final_metrics = ""

    try:
        if "Pipeline" in model_option:
            en_model_name = "LLaMA-3.1-8b (EN)" if "LLaMA" in model_option else "Phi-3.5-Instruct (EN)"
            text_en = GoogleTranslator(source='ro', target='en').translate(document_text)
            model, tokenizer = load_model(en_model_name)
            summary_en = generate_core_summary(model, tokenizer, text_en, PROMPT_EN, en_model_name)
            final_summary = GoogleTranslator(source='en', target='ro').translate(summary_en)
            eval_lang = "ro"

        else:
            model, tokenizer = load_model(model_option)
            lang = MODEL_PATHS[model_option]["lang"]
            active_prompt = PROMPT_RO if lang == "ro" else PROMPT_EN
            final_summary = generate_core_summary(model, tokenizer, document_text, active_prompt, model_option)
            eval_lang = lang

        if reference_text and len(reference_text.strip()) > 0:
            final_metrics = calculate_metrics(final_summary, reference_text, eval_language=eval_lang)
        else:
            final_metrics = "Introduceti un text de referinta pentru a calcula scorurile ROUGE si BERTScore."

    except Exception as e:
        final_summary = f"Eroare la generare: {str(e)}"
        final_metrics = "Eroare la calculul metricilor."

    return final_summary, final_metrics

def main():
    dropdown_options = list(MODEL_PATHS.keys()) + [
        "Pipeline (RO -> Traducere -> LLaMA-3.1 EN -> Traducere RO)",
        "Pipeline (RO -> Traducere -> Phi-3.5 EN -> Traducere RO)"
    ]

    with gr.Blocks(title="Asistent Juridic Evaluare Modele") as gui_interface:
        gr.Markdown("# Evaluator Modele AI - Sumarizare Juridica (Licenta)")

        with gr.Row():
            with gr.Column(scale=1):
                input_document = gr.Textbox(lines=8, label="1. Document Sursa (Text legislativ)")
                input_reference = gr.Textbox(lines=4, label="2. Rezumat de Referinta (Pentru calcul metrici)")

                selector_model = gr.Dropdown(
                    choices=dropdown_options,
                    value="RoLLaMA-3.1-8b (RO)",
                    label="3. Selecteaza Modelul / Arhitectura"
                )
                btn_execute = gr.Button("Genereaza si Evalueaza", variant="primary")

            with gr.Column(scale=1):
                output_summary = gr.Textbox(lines=8, label="Rezumat Generat de AI", interactive=False)
                output_metrics = gr.Textbox(lines=8, label="Performanta (ROUGE & BERTScore)", interactive=False)

        btn_execute.click(
            fn=process_request,
            inputs=[input_document, input_reference, selector_model],
            outputs=[output_summary, output_metrics],
            show_progress="full"
        )

    gui_interface.launch(share=True, debug=False)

if __name__ == "__main__":
    main()