from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.src_lang = "eng_Latn"

def translate(text):

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva")
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


interface = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(label="Enter English Text"),
    outputs=gr.Textbox(label="Hindi Translation"),
    title="English → Hindi Translator",
)

interface.launch()