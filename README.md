# YALENLP @ **PerAnsSumm 2025**
This repository contains the code and experiments from YaleNLP Team for [**PerAnsSumm 2025**](https://peranssumm.github.io/docs/), a shared task on **perspective-aware summarization**, as part of **NAACL 2025**.  
We conduct various evaluations using **LLama 3.3 70B**, **GPT-4o**, and other models, experimenting with different settings, prompting strategies, and fine-tuning methods.

---

## **🧪 Experiment Checklist**

### **1️⃣ Zero-shot Prompting**
- [ ] **LLama 3.3 70B Zero-shot**
  - [x] Task A w/ simple prompting
  - [x] Task A w/ enhanced prompting
  - [x] Task B w/ simple prompting
  - [x] Task B w/ enhanced prompting
  - [ ] Test **various hyperparameters** (temperature, max_len, top_p, etc.)
- [ ] **GPT-4o Zero-shot**
  - [ ] Task A 
  - [ ] Task B
  - [ ] Compare **prompting strategies** with LLama 3.3 70B

### **2️⃣ Few-shot Prompting**
- [ ] **LLama 3.3 70B Few-shot**
  - [ ] **3-shot**
  - [ ] **5-shot**
  - [ ] **Sentence-transformer based few-shot case selection**
- [ ] **GPT-4o Few-shot**
  - [ ] **3-shot**
- [ ] **Mixture of Agents**
  - [ ] Test **various settings** for combining multiple agents


### **3️⃣ Supervised Fine-tuning**
- [ ] **Fine-tuning for Task A & B**
  - [ ] Explore **various fine-tuning methods**
- [ ] **BERT Model Fine-tuning for Task A**
  - [ ] Train **BERT for Span Identification & Classification**
  - [ ] Use **BIO tagging** for dataset construction
  - [ ] Evaluate **results & limitations**
- [ ] **Summarize findings**
  - [ ] Possibly **write a short appendix** on **BERT NER results & limitations** (e.g., "The results were really bad...")
