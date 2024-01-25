import tensorflow as tf
from Train import *
import numpy

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(15, activation='relu', input_shape=(9,)))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))

model.add(tf.keras.layers.Dense(9, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
def train() :
    for i in range(len(X)) :
        jeu = tf.expand_dims(tf.convert_to_tensor(X[i]), axis=0)
        rep = tf.expand_dims(tf.convert_to_tensor(Y[i]), axis=0)
        for epoch in range(100):
            history = model.fit(jeu, rep, epochs=1, verbose=0)

            pourcent = (i * 500 + epoch) / (len(X) * 500)
            nb_eq = int(pourcent * 10)
            progress_bar = '=' * nb_eq + '-' * (10 - nb_eq)

            training_status = f"Entraînement {i+1}/{len(X)} ({epoch+1}/100)"
            output = f"|[{progress_bar}]| {int(pourcent*100)}% || {training_status}"
                        
            print(f"\r\033[K{output}", end='')

    model.save("IA.keras")

train()
model.load_weights("IA.keras")
print(model.predict(tf.convert_to_tensor([[1,-1,0, 0,-1,0, 0,1,0]])))

