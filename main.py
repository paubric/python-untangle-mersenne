# Generating data
import random
import matplotlib.pyplot as plt
import numpy as np

sample_count = 100
training_seed_count = 100

random_series = np.zeros((training_seed_count, sample_count))
test_series = np.zeros((training_seed_count, sample_count))
seeds = np.asarray(range(0, training_seed_count))
test_seeds = np.asarray(range(0, training_seed_count))

for seed in seeds:
    #print('Seed: ', seed)
    random.seed(seed)
    new_sample_list = np.zeros((None))
    for sample in range(0, sample_count-1):
        new_sample_list = np.append(new_sample_list, random.random())
        #print(sample)
    random_series[seed] = new_sample_list

for seed in seeds:
    #print('Seed: ', seed)
    random.seed(seed)
    for i in range(0, sample_count):
        random.random()
    test_sample_list = np.zeros((None))
    for sample in range(0, sample_count-1):
        test_sample_list = np.append(test_sample_list, random.random())
        #print(sample)
    test_series[seed] = test_sample_list

print(random_series)
print(test_series)

# Training and testing
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

batch_size = 1000
num_classes = 10
epochs = 1000

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(sample_count,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(random_series, seeds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

predictions = model.predict(test_series)
print(predictions)

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Untangle Mersenne')
plt.plot(seeds)
plt.plot(predictions)
ax.set_xlabel('Ground-truth Seed')
ax.set_ylabel('Predicted Seed')
plt.legend(['Ground-truth Seeds', 'Predicted Seeds'])
plt.show()
