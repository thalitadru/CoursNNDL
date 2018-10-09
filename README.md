# Du perceptron à l'apprentissage profond
Séances préparés pour le cours de Neurosciences computationnelles, Master Sciences Cognitives de l'Université de Bordeaux

## Plan de Cours
### Séance 1
1. Perceptron ~ 15 min
    1. Le neurone biologique
    2. Le perceptron
    3. Logique avec un Perceptron
        * Exercice XOR

2. Intro à l'apprentissage automatique ~ 15 min
    1. Types d'apprentissage
    2. Apprentissage supervisé
        * **Pratique** playground.tensorflow.org

3. Principes d'optimisation par descente du gradient ~ 1h
    * **Pratique** Régression linéaire avec numpy

4. Perceptron multi-couches ~ 1h30
    * **Pratique** Introduction à pytorch
    * **Pratique** Forward pass MLP en numpy et pytorch

### Séance 2
5. La retro-propagation des gradients ~ 1h30
    * **Pratique** MLP en pytorch avec autograd
    * **Pratique** MLP en pytorch avec `nn` et `optim`

6. Réseaux convolutifs profonds et le cortex visuel ~ 1h30
    1. Cortex visuel
        1. Chemin ventral et dorsal
        1. Cortex visuel primaire    
            1. retinotopie
            2. cellules simples et complexes
        3. Détection de catégories d'objets dans l'inferotemporel
    2. Réseaux convolutifs
        1. Historique
        2. Éléments et relation avec le cortex V1
            1. Convolution
            2. Pooling (sous échantillonnage)
            3. Non-linéarité
        3. Difficultés d’entraînement
        4. Interpretabilité
        5. Examples adversaires
    * **Pratique** tutoriel NN pytorch CNN + Cifar 10
