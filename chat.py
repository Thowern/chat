from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Carica il modello e il tokenizer
model = TFAutoModelForCausalLM.from_pretrained('./modello_finetuned')
tokenizer = AutoTokenizer.from_pretrained('./modello_finetuned', use_fast=True)

def genera_risposta(input_text, max_length=100):
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    input_length = tf.shape(input_ids)[1]
    output = model.generate(
        input_ids,
        max_length=input_length + max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=4,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    risposta = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    return risposta.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    input_text = data.get('message', '')
    risposta = genera_risposta(input_text)
    return jsonify({'response': risposta})

if __name__ == '__main__':
    app.run(debug=True)
