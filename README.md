# Du perceptron à l'apprentissage profond
Séances préparés pour le cours de Neurosciences computationnelles, Master Sciences Cognitives de l'Université de Bordeaux

## Plan de Cours
### Séance 1
1. Perceptron
    1. Le neurone biologique
    2. Le perceptron
    3. Logique avec un Perceptron
        * Exercice XOR

2. Intro à l'apprentissage automatique
    1. Types d'apprentissage
    2. Apprentissage supervisé
        * **Pratique** playground.tensorflow.org

3. Principes d'optimisation par descente du gradient
    * **Pratique** Régression linéaire avec numpy

### Séance 2

4. Perceptron multi-couches
    * **Pratique** Forward pass MLP en numpy et pytorch

5. La retro-propagation des gradients
    * **Pratique** MLP en pytorch avec autograd
    * **Pratique** MLP en pytorch avec `nn` et `optim` [extra]

6. Réseaux convolutifs profonds et le cortex visuel
    1. Cortex visuel
        1. Chemin ventral et dorsal
        2. Cortex visuel primaire    
            1. retinotopie
            2. cellules simples et complexes
        3. Détection de catégories d'objets dans l'inferotemporel
    2. Réseaux convolutifs
        1. Éléments et relation avec le cortex V1
            1. Convolution
            2. Pooling (sous échantillonnage)
        2. Difficultés d’entraînement
        3. Interpretabilité
        4. Examples adversaires
    * **Pratique** tutoriel NN pytorch CNN + Cifar 10 [extra] 
