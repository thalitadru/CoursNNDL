#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:06:19 2017

@author: thalita

Tutoriel MNIST

"""

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#%% Parametres
nombre_neurones = 10
taux_apprentissage = 0.5
nombre_iterations = 1000
taille_batch = 100
#%% (Télé)charger les données
mnist = input_data.read_data_sets("data", one_hot=True)
#%% Déclarer les entrées
# images de 28 x 28 pixels, soit un vecteur de 784 positions
images = tf.placeholder(tf.float32, [None, 784])
# sorties attendues : un vecteur avec 10 positions, chaqu'une indique une chiffre
sortie_attendue = tf.placeholder(tf.float32, [None, 10])

#%% Déclarer une couche de neurones
sortie = tf.layers.dense(
            images, nombre_neurones,
            activation=tf.nn.tanh,
            use_bias=True,
            name='sortie')

#%% Definir une mesure de pertes
loss = tf.losses.softmax_cross_entropy(sortie_attendue, sortie)

#%% Definir une metrique de taux d'erreur (Accuracy)
correct_prediction = tf.equal(tf.argmax(sortie_attendue, 1),
                              tf.argmax(sortie, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% Definir une procedure d'optimization
optimizer = tf.train.GradientDescentOptimizer(taux_apprentissage)
ajustement_poids = optimizer.minimize(loss)

#%% Créer un session Tensorflow pour executer les calculs
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
#%% Entrainer le modèle
for i in range(nombre_iterations):
    batch_xs, batch_ys = mnist.train.next_batch(taille_batch)
    _, loss_val = sess.run([ajustement_poids, loss],
         feed_dict={images: batch_xs, sortie_attendue: batch_ys})
    print('training ', i, 'loss: ', loss_val,
          end='\r', flush=True)

#%% Tester le modèle sur des nouvelles images
print('\n test accuracy: ',
      sess.run(accuracy, feed_dict={images: mnist.test.images,
                                    sortie_attendue: mnist.test.labels}))

#%%
sess.close()