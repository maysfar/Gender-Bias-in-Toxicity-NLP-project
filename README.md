# Mitigating Gender Bias in Toxicity Detection

This project investigates **gender-related bias in toxicity classification models** and proposes two mitigation strategies: **gender masking** and **gender swapping**.  
We fine-tune transformer architectures (BERT and RoBERTa) on the [Jigsaw Unintended Bias in Toxicity Classification dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) and evaluate them using both standard performance metrics and fairness-oriented measures.  

Our experiments show that:
- **Bias is present even if small**, and can affect toxicity predictions for different gender subgroups.  
- **Masking** (replacing gendered words with a neutral placeholder) consistently improves fairness across both BERT and RoBERTa.  
- **Swapping** (adding counterfactual gender-flipped sentences) significantly improves BERT’s robustness while maintaining accuracy.  
- Importantly, **fine-tuning on augmented datasets did not reduce accuracy**.  

---

## Resources

### Models
- [moatazhamza194/tc-bert](https://huggingface.co/moatazhamza194/tc-bert)  
- [moatazhamza194/tc-roberta](https://huggingface.co/moatazhamza194/tc-roberta)  
- [moatazhamza194/tc-bert_mask](https://huggingface.co/moatazhamza194/tc-bert_mask)  
- [moatazhamza194/tc-roberta_mask](https://huggingface.co/moatazhamza194/tc-roberta_mask)  
- [moatazhamza194/tc-bert_swap](https://huggingface.co/moatazhamza194/tc-bert_swap)  
- [moatazhamza194/tc-roberta_swap](https://huggingface.co/moatazhamza194/tc-roberta_swap)  
- [moatazhamza194/gender_classification-deberta](https://huggingface.co/moatazhamza194/gender_classification-deberta)  

### Datasets
- [moatazhamza194/gb_train](https://huggingface.co/datasets/moatazhamza194/gb_train)  
- [moatazhamza194/gb_test](https://huggingface.co/datasets/moatazhamza194/gb_test)  
- [moatazhamza194/gb_train_mask](https://huggingface.co/datasets/moatazhamza194/gb_train_mask)  
- [moatazhamza194/gb_train_swap](https://huggingface.co/datasets/moatazhamza194/gb_train_swap)  
- [moatazhamza194/gb_test_gendered](https://huggingface.co/datasets/moatazhamza194/gb_test_gendered)  

---

## Repository Structure

- **`pipeline/`**  
  Google Colab notebooks containing the full experimental pipeline, from preprocessing and augmentation to training and evaluation.  

- **`src/`**  
  Python modules implementing the functionality used inside the notebooks.  

- **`data/`**  
  Contains the **gendered words dictionary** (used for masking and swapping) as well as the dataset for **gender classification**.  

- **`prompts/`**  
  All prompts used during the project to generate synthetic data, counterfactual examples, and gendered word dictionaries with large language models.  

---

## Citation

If you use this repository, please cite the models and datasets linked above.
