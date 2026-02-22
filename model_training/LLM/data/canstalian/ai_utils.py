from transformers import pipeline

def generate_pentest_report(target_info):
    # Initialize the text generation pipeline
    generator = pipeline("text-generation", model="gpt2")
    
    # Generate a report based on the target information
    prompt = f"Generate a penetration testing report for the following target and findings:\n\n{target_info}\n\nReport:"
    generated_text = generator(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
    
    # Extract the generated report (remove the prompt)
    report = generated_text.split("Report:")[1].strip()
    
    return report
