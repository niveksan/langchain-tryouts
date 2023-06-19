from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length= 60)
print(tokenizer.decode(outputs[0]))