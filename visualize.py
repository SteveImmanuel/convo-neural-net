import matplotlib.pyplot as plt
from coolcnn.models import Sequential

model = Sequential.load('cnn-good.model')

history = model.history
print(len(history))
mse = [x[0] for x in history]
acc = [x[1]/163 for x in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(mse)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')

ax2.plot(acc)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

plt.suptitle('Training Result', fontsize=20)
plt.show()