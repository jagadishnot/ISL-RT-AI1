import matplotlib.pyplot as plt

train_loss = [340,310,280,250]
val_loss = [330,305,270,240]

plt.plot(train_loss,label="Train Loss")
plt.plot(val_loss,label="Validation Loss")

plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Curve")

plt.show()