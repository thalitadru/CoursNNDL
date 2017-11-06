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
taux_apprentissage = 1e-4
nombre_iterations = 1000
taille_batch = 50
proba_dropout = 0.5

#%% (Télé)charger les données
mnist = input_data.read_data_sets("data", one_hot=True)

#%% Déclarer les entrées
# images de 28 x 28 pixels, soit un vecteur de 784 positions
images = tf.placeholder(tf.float32, [None, 784])
# sorties attendues : un vecteur avec 10 positions, chaqu'une indique une chiffre
sortie_attendue = tf.placeholder(tf.float32, [None, 10])

#%% Déclarer un reseau CNN
# reorganizer vecteur en forme de image de 28 par 28 pixels
images_carres = tf.reshape(images, [-1, 28, 28, 1])
conv1 = tf.layers.conv2d(images_carres,
                         filters=32,
                         kernel_size=5,
                         name='conv1')

pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

conv2 = tf.layers.conv2d(pool1,
                         filters=64,
                         kernel_size=5,
                         name='conv2')

pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

# on va retourner au format vecteur
# pour passer à des couches denses traditionelles
vecteur = tf.contrib.layers.flatten(pool2)

# Couches denses traditionelles
dense3 = tf.layers.dense(vecteur,
                          units=1024,
                          activation=tf.nn.relu,
                          name='dense3')

# couche dropout pour eviter le surentrainement
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
dropout3 = tf.layers.dropout(dense3, rate=keep_prob, training=is_training,
                             name='dropout3')
# couche de sortie
sortie = tf.layers.dense(dropout3,
                         units=10,
                         activation=tf.nn.relu,
                         name='sortie')

#%% Definir une mesure de pertes
loss = tf.losses.softmax_cross_entropy(sortie_attendue, sortie)

#%% Definir une metrique de taux d'erreur (Accuracy)
correct_prediction = tf.equal(tf.argmax(sortie_attendue, 1),
                              tf.argmax(sortie, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% Definir une procedure d'optimization
optimizer = tf.train.AdamOptimizer(taux_apprentissage)
ajustement_poids = optimizer.minimize(loss)

#%% Créer un session Tensorflow pour executer les calculs
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
#%% Entrainer le modèle
for i in range(nombre_iterations):
    batch_xs, batch_ys = mnist.train.next_batch(taille_batch)
    _, loss_val = sess.run([ajustement_poids, loss],
         feed_dict={images: batch_xs, sortie_attendue: batch_ys,
                    keep_prob: proba_dropout, is_training: True })
    print('training ', i, 'loss: ', loss_val,
          end='\r', flush=True)

#%% Tester le modèle sur des nouvelles images
print('\n test accuracy: ',
      sess.run(accuracy, feed_dict={images: mnist.test.images,
                                    sortie_attendue: mnist.test.labels,
                                    keep_prob: proba_dropout,
                                    is_training: False}))

#%%
sess.close()