import tensorflow as tf
import numpy as np

jeu = [0,0,0, 0,0,0, 0,0,0]
gagnant = None

IA = tf.keras.models.load_model('IA.keras')

def update_weights(jeu, reward):
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)
    loss = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        prediction = IA(tf.expand_dims(tf.convert_to_tensor(jeu), axis=0))
        loss_value = loss(reward, prediction)
  
        gradients = tape.gradient(loss_value, IA.trainable_variables)
        IA.optimizer.apply_gradients(zip(gradients, IA.trainable_variables))


def gagne(jeu):
    win_states = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]

    if any(jeu[i] == jeu[j] == jeu[k] == 1 for i, j, k in win_states):
        update_weights(jeu,1)
        IA.save("IA.keras")
        return "IA gagne"

    if any(jeu[i] == jeu[j] == jeu[k] == -1 for i, j, k in win_states):
        update_weights(jeu,-1)
        IA.save("IA.keras")
        return "Joueur gagne"

    if (sum(jeu) == 1 or sum(jeu) == -1) and 0 not in jeu :
        update_weights(jeu,1)
        IA.save("IA.keras")
        return "Match null"

    return None

def jouer_ia(pos,jeu) :
    loca = np.argmax(pos)
    tmp_jeu = jeu.copy()
    calc = 0

    while jeu[loca] == 1 or jeu[loca] == -1 :
        tmp_jeu[loca] = 1
        if calc == 5 :
            update_weights(tmp_jeu,-1)
            calc = 0
            tmp_jeu = jeu.copy()
        pos = IA.predict(tf.expand_dims(tf.convert_to_tensor(jeu), axis=0))
        loca = np.argmax(pos)
        calc+=1

    jeu[loca] = 1
    update_weights(jeu,1)
    return jeu

def jouer_joueur(pos,jeu) :
    while jeu[pos] == 1 or jeu[pos] == -1 : 
        pos = int(input("Entrer une valeur entre 0 et 8 : "))
    jeu[pos] = -1
    return jeu

def afficher_jeu(jeu):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for i in range(3):
        for j in range(3):
            print(symbols[jeu[3*i + j]], end=' ')
            if j < 2:
                print('|', end=' ')
        print()
        if i < 2:
            print('-'*9)


while not gagnant :
    pos = int(input("Entrer la ou vous vouler jouer : "))
    while pos > 8 or pos < 0:
        pos = int(input("Entrer une valeur entre 0 et 8 : "))
    jeu = jouer_joueur(pos,jeu)

    gagne(jeu)
    afficher_jeu(jeu)
    print("\n\n")

    pos_ia = IA.predict(tf.expand_dims(tf.convert_to_tensor(jeu), axis=0))
    jeu = jouer_ia(pos_ia,jeu)

    gagne(jeu)
    afficher_jeu(jeu)
    print("\n\n")


print(gagnant)


