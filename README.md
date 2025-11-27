# Galaxy-Morphology-and-Simulation-Bias-Analysis-Research

\section*{PROJECT OVERVIEW}
\subsection*{Trustworthy AI for Galaxy Morphology Classification and Bias Mitigation}

This repository contains the codebase, datasets, and evaluation pipeline for a research project conducted in the University of Michigan’s Department of Astronomy. The work focuses on developing a trustworthy AI framework for galaxy morphology classification, quantifying labeling bias, and correcting resolution-driven distortions in astronomical datasets.

The project leverages a ResNet-50 convolutional neural network to classify more than 39,000 galaxy images (spiral vs.\ elliptical), identifies systematic biases in human-provided labels, and applies modified deep-learning loss functions to mitigate these biases. Using both real and simulated galaxies, the project improves classification consistency and enhances fairness across varying observational conditions.

\section*{TASKS}

\subsection*{1. Galaxy Classification + Bias Quantification}

\subsubsection*{Key Points}

\begin{itemize}
    \item \textbf{Galaxy Morphology Classification:} Trained a ResNet-50 model on 39k+ galaxy images, achieving over 80\% accuracy. Analyzed training/validation behavior using learning curves and activation maps.
    \item \textbf{Bias Measurement:} Implemented Bias2014 (bin-variance) and Bias2018 (KL-divergence) to quantify how image resolution, angular size, and PSF influence human-labeled classifications.
    \item \textbf{Resolution-Driven Bias:} Identified strong bias patterns where distant, low-resolution spiral galaxies are mislabeled as featureless ellipticals.
    \item \textbf{Visualization:} Generated ROC curves, PR curves, accuracy–bias tradeoff plots, and Grad-CAM interpretability heatmaps.
\end{itemize}

Detailed background and scientific context are provided in the MIDAS Summit presentation and NSF proposal included in this repository.

\subsection*{2. Debiasing Framework}

\subsubsection*{Key Points}

\begin{itemize}
    \item \textbf{Modified Loss Function:} Incorporated a probabilistic bias-likelihood term into the CNN loss, enabling the model to discount misleading biased labels.
    \item \textbf{Bias Reduction:} Achieved approximately 65\% reduction in resolution-driven labeling bias compared to the baseline CNN model.
    \item \textbf{Improved Model Consistency:} Enhanced classification stability by more than 20\% across varying observational regimes using both real and simulated galaxies.
    \item \textbf{Validation Pipeline:} Built a GPU-accelerated simulation pipeline using 100k+ synthetic galaxies to verify classifier robustness.
\end{itemize}

\subsection*{3. Repository Contents}

\subsubsection*{Includes:}

\begin{itemize}
    \item CNN training scripts for baseline and debiased models
    \item Bias computation tools (sweeps, metrics, statistical summaries)
    \item Visualization scripts (ROC, PR, tradeoff curves, learning curves)
    \item Activation map and interpretability utilities (Grad-CAM)
    \item Simulation validation scripts
    \item Data preprocessing and filtering tools
    \item Intermediate outputs (CSV, logs, PDF figures)
\end{itemize}

\section*{SCIENTIFIC IMPACT}

This repository contributes to the emerging domain of \emph{trustworthy AI for astronomy}, enabling:
\begin{itemize}
    \item systematic identification of human labeling biases,
    \item principled debiasing of deep-learning models,
    \item improved morphological classification accuracy under varied observational conditions,
    \item scalable frameworks for mapping imaging pixels to astrophysical properties.
\end{itemize}

The techniques also generalize to other remote-sensing fields where expert labels contain hidden systematic bias.

\section*{CONTACT}

For questions regarding methodology, reproducibility, or collaboration:
\begin{itemize}
    \item University of Michigan -- Department of Astronomy
    \item Galaxy Morphology \& Simulation Bias Group
\end{itemize}
