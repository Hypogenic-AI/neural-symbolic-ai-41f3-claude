# Downloaded Papers

This directory contains 20 key papers on Neural-Symbolic (NeSy) learning, downloaded from arXiv. Papers were selected from a relevance-ranked set of 121 results (paper-finder, diligent mode) to cover foundational methods, major surveys, and recent LLM-era work. Full ranked list with abstracts: `../paper_search_results/ranked_papers.json`.

### 1. The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision
- **File**: `nscl_concept_learner_2019.pdf`
- **Authors**: Jiayuan Mao, Chuang Gan, Pushmeet Kohli, J. Tenenbaum, Jiajun Wu
- **Year**: 2019  |  **Citations (approx)**: 846
- **Abstract**: We propose the Neuro-Symbolic Concept Learner (NS-CL), a model that learns visual concepts, words, and semantic parsing of sentences without explicit supervision on any of them; instead, our model learns by simply looking at images and reading paired questions and answers. Our model builds an object-based scene representation and translates sentences into executable, symbolic programs. To bridge the learning of two modules, we use a neuro-symbolic reasoning module that executes these programs on the latent scene representation. Analogical to human concept learning, the perception module learns...

### 2. DeepProbLog: Neural Probabilistic Logic Programming
- **File**: `deepproblog_2018.pdf`
- **Authors**: Robin Manhaeve, Sebastijan Dumancic, A. Kimmig, T. Demeester, L. D. Raedt
- **Year**: 2018  |  **Citations (approx)**: 748
- **Abstract**: We introduce DeepProbLog, a probabilistic logic programming language that incorporates deep learning by means of neural predicates. We show how existing inference and learning techniques can be adapted for the new language. Our experiments demonstrate that DeepProbLog supports both symbolic and subsymbolic representations and inference, 1) program induction, 2) probabilistic (logic) programming, and 3) (deep) learning from examples. To the best of our knowledge, this work is the first to propose a framework where general-purpose neural networks and expressive probabilistic-logical modeling and...

### 3. Learning Explanatory Rules from Noisy Data
- **File**: `dilp_explanatory_rules_2017.pdf`
- **Authors**: Richard Evans, Edward Grefenstette
- **Year**: 2017  |  **Citations (approx)**: 553
- **Abstract**: Artificial Neural Networks are powerful function

approximators capable of modelling solutions to

a wide variety of problems, both supervised and

unsupervised. As their size and expressivity increases,

so too does the variance of the model,

yielding a nearly ubiquitous overfitting problem.

Although mitigated by a variety of model regularisation

methods, the common cure is to seek

large amounts of training data—which is not necessarily

easily obtained—that sufficiently approximates

the data distribution of the domain we wish

to test on. In contrast, logic programming methods

such as ...

### 4. Neurosymbolic AI: the 3rd wave
- **File**: `neurosymbolic_3rd_wave_2020.pdf`
- **Authors**: A. Garcez, L. Lamb
- **Year**: 2020  |  **Citations (approx)**: 531
- **Abstract**: Current advances in Artificial Intelligence (AI) and Machine Learning have achieved unprecedented impact across research communities and industry. Nevertheless, concerns around trust, safety, interpretability and accountability of AI were raised by influential thinkers. Many identified the need for well-founded knowledge representation and reasoning to be integrated with deep learning and for sound explainability. Neurosymbolic computing has been an active area of research for many years seeking to bring together robust learning in neural networks with reasoning and explainability by offering ...

### 5. Neural-Symbolic Computing: An Effective Methodology for Principled Integration of Machine Learning and Reasoning
- **File**: `nesy_computing_methodology_2019.pdf`
- **Authors**: A. Garcez, M. Gori, L. Lamb, L. Serafini, Michael Spranger, S. Tran
- **Year**: 2019  |  **Citations (approx)**: 383
- **Abstract**: Current advances in Artificial Intelligence and machine learning in general, and deep learning in particular have reached unprecedented impact not only across research communities, but also over popular media channels. However, concerns about interpretability and accountability of AI have been raised by influential thinkers. In spite of the recent impact of AI, several works have identified the need for principled knowledge representation and reasoning mechanisms integrated with deep learning-based systems to provide sound and explainable models for such systems. Neural-symbolic computing aims...

### 6. Logic Tensor Networks
- **File**: `logic_tensor_networks_2020.pdf`
- **Authors**: Samy Badreddine, A. Garcez, L. Serafini, M. Spranger
- **Year**: 2020  |  **Citations (approx)**: 345
- **Abstract**: Artificial Intelligence agents are required to learn from their surroundings and to reason about the knowledge that has been learned in order to make decisions. While state-of-the-art learning from data typically uses sub-symbolic distributed representations, reasoning is normally useful at a higher level of abstraction with the use of a first-order logic language for knowledge representation. As a result, attempts at combining symbolic AI and neural computation into neural-symbolic systems have been on the increase. In this paper, we present Logic Tensor Networks (LTN), a neurosymbolic formal...

### 7. NeurASP: Embracing Neural Networks into Answer Set Programming
- **File**: `neurasp_2020.pdf`
- **Authors**: Zhun Yang, Adam Ishay, Joohyung Lee
- **Year**: 2020  |  **Citations (approx)**: 214
- **Abstract**: We present NeurASP, a simple extension of answer set programs by embracing neural networks. By treating the neural network output as the probability distribution over atomic facts in answer set programs, NeurASP provides a simple and effective way to integrate sub-symbolic and symbolic computation. We demonstrate how NeurASP can make use of a pre-trained neural network in symbolic computation and how it can improve the neural network's perception result by applying symbolic reasoning in answer set programming. Also, NeurASP can make use of ASP rules to train a neural network better so that a n...

### 8. From statistical relational to neurosymbolic artificial intelligence: A survey
- **File**: `starai_to_nesy_survey_2021.pdf`
- **Authors**: Giuseppe Marra, Sebastijan Dumančić, Robin Manhaeve, L. D. Raedt
- **Year**: 2021  |  **Citations (approx)**: 119
- **Abstract**: This survey explores the integration of learning and reasoning in two different fields of artificial intelligence: neurosymbolic and statistical relational artificial intelligence. Neurosymbolic artificial intelligence (NeSy) studies the integration of symbolic reasoning and neural networks, while statistical relational artificial intelligence (StarAI) focuses on integrating logic with probabilistic graphical models. This survey identifies seven shared dimensions between these two subfields of AI. These dimensions can be used to characterize different NeSy and StarAI systems. They are concerne...

### 9. Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding
- **File**: `ns_vqa_2018.pdf`
- **Authors**: Kexin Yi, Jiajun Wu, Chuang Gan, A. Torralba, Pushmeet Kohli, J. Tenenbaum
- **Year**: 2018  |  **Citations (approx)**: 708
- **Abstract**: We marry two powerful ideas: deep representation learning for visual recognition and language understanding, and symbolic program execution for reasoning. Our neural-symbolic visual question answering (NS-VQA) system first recovers a structural scene representation from the image and a program trace from the question. It then executes the program on the scene representation to obtain an answer. Incorporating symbolic structure as prior knowledge offers three unique advantages. First, executing programs on a symbolic space is more robust to long program traces; our model can solve complex reaso...

### 10. A Semantic Loss Function for Deep Learning with Symbolic Knowledge
- **File**: `semantic_loss_2017.pdf`
- **Authors**: Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, Guy Van den Broeck
- **Year**: 2017  |  **Citations (approx)**: 546
- **Abstract**: This paper develops a novel methodology for using symbolic knowledge in deep learning. From first principles, we derive a semantic loss function that bridges between neural output vectors and logical constraints. This loss function captures how close the neural network is to satisfying the constraints on its output. An experimental evaluation shows that it effectively guides the learner to achieve (near-)state-of-the-art results on semi-supervised multi-class classification. Moreover, it significantly increases the ability of the neural network to predict structured objects, such as rankings a...

### 11. Scallop: A Language for Neurosymbolic Programming
- **File**: `scallop_lang_2023.pdf` (PLDI 2023 language/system paper; precursor is the NeurIPS 2021 "Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning")
- **Authors**: Ziyang Li, Jiani Huang, Mayur Naik (and colleagues)
- **Year**: 2023  |  **Citations (approx)**: 89 (line of work)

### 12. NeuPSL: Neural Probabilistic Soft Logic
- **File**: `neupsl_2022.pdf`
- **Authors**: Connor Pryor, Charles Dickens, Eriq Augustine, Alon Albalak, William Wang, L. Getoor
- **Year**: 2022  |  **Citations (approx)**: 49
- **Abstract**: In this paper, we introduce Neural Probabilistic Soft Logic (NeuPSL), a novel neuro-symbolic (NeSy) framework that unites state-of-the-art symbolic reasoning with the low-level perception of deep neural networks. To model the boundary between neural and symbolic representations, we propose a family of energy-based models, NeSy Energy-Based Models, and show that they are general enough to include NeuPSL and many other NeSy approaches. Using this framework, we show how to seamlessly integrate neural and symbolic parameter learning and inference in NeuPSL. Through an extensive empirical evaluatio...

### 13. Deep Symbolic Learning: Discovering Symbols and Rules from Perceptions
- **File**: `deep_symbolic_learning_2022.pdf`
- **Authors**: Alessandro Daniele, Tommaso Campari, Sagar Malhotra, L. Serafini
- **Year**: 2022  |  **Citations (approx)**: 27
- **Abstract**: Neuro-Symbolic (NeSy) integration combines symbolic reasoning with Neural Networks (NNs) for tasks requiring perception and reasoning. Most NeSy systems rely on continuous relaxation of logical knowledge, and no discrete decisions are made within the model pipeline. Furthermore, these methods assume that the symbolic rules are given. In this paper, we propose Deep Symboilic Learning (DSL), a NeSy system that learns NeSy-functions, i.e., the composition of a (set of) perception functions which map continuous data to discrete symbols, and a symbolic function over the set of symbols. DSL simultan...

### 14. Softened Symbol Grounding for Neuro-symbolic Systems
- **File**: `softened_symbol_grounding_2024.pdf`
- **Authors**: Zenan Li, Yuan Yao, Taolue Chen, Jingwei Xu, Chun Cao, Xiaoxing Ma, Jian Lü
- **Year**: 2024  |  **Citations (approx)**: 22
- **Abstract**: Neuro-symbolic learning generally consists of two separated worlds, i.e., neural network training and symbolic constraint solving, whose success hinges on symbol grounding, a fundamental problem in AI. This paper presents a novel, softened symbol grounding process, bridging the gap between the two worlds, and resulting in an effective and efficient neuro-symbolic learning framework. Technically, the framework features (1) modeling of symbol solution states as a Boltzmann distribution, which avoids expensive state searching and facilitates mutually beneficial interactions between network traini...

### 15. LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers
- **File**: `linc_2023.pdf`
- **Authors**: Theo X. Olausson, Alex Gu, Benjamin Lipkin, Cedegao E. Zhang, Armando Solar-Lezama, Josh Tenenbaum, Roger Levy
- **Year**: 2023  |  **Citations (approx)**: 258
- **Abstract**: Logical reasoning, i.e., deductively inferring the truth value of a conclusion from a set of premises, is an important task for artificial intelligence with wide potential impacts on science, mathematics, and society. While many prompting-based strategies have been proposed to enable Large Language Models (LLMs) to do such reasoning more effectively, they still appear unsatisfactory, often failing in subtle and unpredictable ways. In this work, we investigate the validity of instead reformulating such tasks as modular neurosymbolic programming, which we call LINC: Logical Inference via Neurosy...

### 16. Towards Data-And Knowledge-Driven AI: A Survey on Neuro-Symbolic Computing
- **File**: `data_knowledge_driven_survey_2022.pdf`
- **Authors**: Wenguan Wang, Yi Yang, Fei Wu
- **Year**: 2022  |  **Citations (approx)**: 65
- **Abstract**: Neural-symbolic computing (NeSy), which pursues the integration of the symbolic and statistical paradigms of cognition, has been an active research area of Artificial Intelligence (AI) for many years. As NeSy shows promise of reconciling the advantages of reasoning and interpretability of symbolic representation and robust learning in neural networks, it may serve as a catalyst for the next generation of AI. In the present paper, we provide a systematic overview of the recent developments and important contributions of NeSy research. First, we introduce study history of this area, covering ear...

### 17. pix2rule: End-to-end Neuro-symbolic Rule Learning
- **File**: `pix2rule_2021.pdf`
- **Authors**: Nuri Cingillioglu, A. Russo
- **Year**: 2021  |  **Citations (approx)**: 12
- **Abstract**: Humans have the ability to seamlessly combine low-level visual input with high-level symbolic reasoning often in the form of recognising objects, learning relations between them and applying rules. Neuro-symbolic systems aim to bring a unifying approach to connectionist and logic-based principles for visual processing and abstract reasoning respectively. This paper presents a complete neuro-symbolic method for processing images into objects, learning relations and logical rules in an end-to-end fashion. The main contribution is a differentiable layer in a deep learning architecture from which ...

### 18. Neurosymbolic Reinforcement Learning and Planning: A Survey
- **File**: `nesy_rl_planning_survey_2023.pdf`
- **Authors**: Kamal Acharya, Waleed Raza, Carlos Dourado, Alvaro Velasquez, Houbing Song
- **Year**: 2023  |  **Citations (approx)**: 54
- **Abstract**: The area of neurosymbolic artificial intelligence (Neurosymbolic AI) is rapidly developing and has become a popular research topic, encompassing subfields, such as neurosymbolic deep learning and neurosymbolic reinforcement learning (Neurosymbolic RL). Compared with traditional learning methods, Neurosymbolic AI offers significant advantages by simplifying complexity and providing transparency and explainability. Reinforcement learning (RL), a long-standing artificial intelligence (AI) concept that mimics human behavior using rewards and punishment, is a fundamental component of Neurosymbolic ...

### 19. Improving Rule-based Reasoning in LLMs using Neurosymbolic Representations
- **File**: `improving_rule_reasoning_llm_2025.pdf`
- **Authors**: Varun Dhanraj, Chris Eliasmith
- **Year**: 2025  |  **Citations (approx)**: 8
- **Abstract**: Large language models (LLMs) continue to face challenges in reliably solving reasoning tasks, particularly those that require precise rule following, as often found in mathematical reasoning. This paper introduces a novel neurosymbolic method that improves LLM reasoning by encoding hidden states into neurosymbolic vectors, enabling problem-solving within a neurosymbolic vector space. The results are decoded and merged with the original hidden state, significantly boosting the model's performance on numerical reasoning tasks. By offloading computation through neurosymbolic representations, this...

### 20. Scalable Neural-Probabilistic Answer Set Programming
- **File**: `scalable_neural_prob_asp_2023.pdf`
- **Authors**: Arseny Skryagin, Daniel Ochs, D. Dhami, K. Kersting
- **Year**: 2023  |  **Citations (approx)**: 22
- **Abstract**: The goal of combining the robustness of neural networks and the expressiveness of symbolic methods has rekindled the interest in Neuro-Symbolic AI. Deep Probabilistic Programming Languages (DPPLs) have been developed for probabilistic logic programming to be carried out via the probability estimations of deep neural networks (DNNs). However, recent SOTA DPPL approaches allow only for limited conditional probabilistic queries and do not offer the power of true joint probability estimation. In our work, we propose an easy integration of tractable probabilistic inference within a DPPL. To this en...
