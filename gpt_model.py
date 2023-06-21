from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

model = AutoModelForCausalLM.from_pretrained("openai-gpt")

input_text = "My day ..."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length= 60)
print(tokenizer.decode(outputs[0]))