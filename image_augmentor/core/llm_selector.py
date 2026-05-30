from transformers import AutoModelForCausalLM, AutoTokenizer

# Load LLM (small demo model, replace with your preferred LLM)
llm_model_name = "NousResearch/Llama-2-7b-hf"  # Example
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")

def llm_suggest_augmentations(class_name, available_ops, top_k=3):
    prompt = f"""
    You are an expert image augmentation policy selector.
    The image belongs to the class: {class_name}.
    Available augmentations are: {", ".join(available_ops)}.
    Suggest the {top_k} most suitable augmentations for this class.
    Only respond with a comma-separated list of augmentation names from the available ones.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    outputs = llm.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    suggestions = [s.strip() for s in response.split(",") if s.strip() in available_ops][:top_k]
    return suggestions