## RAG + Fine Tuning on Medical Dataset
This example starts with a very simple user question:

> *“I have a cough and a high fever. What could it be?”*

In a real system, questions like this are not trivial.  
You usually want the answer to stay calm, consistent, and honest about its limits.  
You also want full control over tone and structure.

---

### Without fine-tuning or RAG

A base LLM can answer this question, but the behavior is unstable.

Typical issues:
- The structure changes between runs  
- The tone is inconsistent  
- Causes are repeated or loosely grouped  
- Serious and mild conditions are mixed  

Nothing is technically wrong, but the output is not really under control.

---

### With fine-tuning and RAG

With fine-tuning, the model learns **how** to respond:
- Same structure every time  
- Calm, cautious tone  
- No overconfident claims  

RAG is then used to add **live operational information**, such as the current on-call doctor.  
This data lives outside the model and can be updated at any time.

The result is a response that is predictable, controlled, and easy to trust.

---

### Why this works

Fine-tuning controls behavior.  
RAG provides fresh facts.

They solve different problems and work best when kept separate.

---

### Under the hood

Base model: Llama 3.2 1B Instruct (4-bit)  
Fine-tuning: QLoRA adapters  
Training data: very small, structure-focused examples  
RAG source: simple external text data  
