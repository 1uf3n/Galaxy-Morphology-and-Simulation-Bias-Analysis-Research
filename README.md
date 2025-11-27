# Galaxy Morphology and Simulation Bias Analysis Research

## PROJECT OVERVIEW

### Trustworthy AI for Galaxy Morphology Classification and Bias Mitigation

This repository contains the full codebase, datasets, and evaluation pipeline for a research project conducted at the **University of Michigan – Department of Astronomy**. The project develops a framework for **trustworthy AI in astronomy**, focused on:

- classifying galaxy morphologies from imaging data,
- identifying systematic labeling bias,
- mitigating resolution-driven distortions,
- validating performance using real and simulated galaxies.

Using a **ResNet-50** classifier trained on **39k+ galaxy images**, we quantify how human-provided labels vary as a function of angular size, PSF, and resolution. A modified loss function integrates a probabilistic bias model, enabling significant bias reduction across observational regimes.

---

## TASKS
### Galaxy Classification + Bias Quantification

- **Galaxy Morphology Classification:**  
  Trained a ResNet-50 CNN achieving **80%+ accuracy** on spiral vs. elliptical classification.  
  Includes tools for learning curves, ROC/PR evaluation, and Grad-CAM interpretability.

- **Bias Measurement (Bias2014 + Bias2018):**  
  Implemented bin-variance and KL-divergence metrics to detect labeling bias w.r.t.  
  $\alpha = r_{\text{angular}} / \text{PSF}$
  revealing strong resolution-dependent misclassification patterns.

- **Observed vs. Intrinsic Distributions:**  
  Quantified labeling drift across bins of physical size, luminosity, and redshift.

- **Visualization Tools:**  
  Bias–accuracy sweeps, PR/ROC curves, learning curves, activation maps.

Detailed scientific context is provided in:  
- **MIDAS Summit Presentation**  
- **NSF AAG 2023 Proposal**

---

### Debiasing Framework

- **Revised Loss Function:**  
  Implemented a bias-aware likelihood term:
  $\mathcal{L} = -\log p(\tilde{y}\mid y, \alpha) - \log p(y\mid x, w)$
  allowing the CNN to discount misleading low-resolution labels.

- **Bias Reduction:**  
  Achieved **$\sim 65\%$ decrease** in resolution-driven labeling bias.

- **Consistency Improvement:**  
  Classification stability improves **$\sim 20\%$** when evaluated on multi-resolution simulations.

- **Simulation-Based Validation:**  
  GPU-accelerated pipeline with **100k+ synthetic galaxies** to stress-test debiased models.

---

## REPOSITORY CONTENTS

- CNN training scripts (baseline + debiased)
- Bias metric computation tools  
- Accuracy–bias sweep visualizers  
- ROC, PR, learning curve generators  
- Grad-CAM activation map utilities  
- Simulation comparison tools  
- Dataset filtering + preprocessing pipelines  
- Intermediate outputs (CSVs, logs, PDFs)

---

## SCIENTIFIC IMPACT

This work contributes to the broader effort to develop **trustworthy AI for astronomy**, demonstrating:

- systematic identification of bias in human-labeled datasets,
- principled debiasing through modified likelihood-based loss functions,
- improved robustness across diverse observational settings,
- scalable pixel-to-physics inference frameworks for galaxy evolution studies.

The methodology generalizes to **remote sensing**, **medical imaging**, and other fields where expert labels contain hidden systematic bias.

---

## CONTACT

For collaboration or questions regarding the method:
**Lufan Wang**  
Email: [lufanw@uw.edu](mailto:lufanw@uw.edu)  
University of Washington
