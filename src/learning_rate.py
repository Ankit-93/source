"""## Learning rate scheduler"""

#We implement a callback that should be called during training to update the learning rate

warmup_step = 4000
class LearningRateScheduler(tf.keras.callbacks.Callback):
    def on_train_batch_start(self, i, batch_logs):
        transformer.optimizer.lr = dk**(-0.5)*min(i**(-0.5),warmup_step**(-3/2)*i)


callback = LearningRateScheduler()

"""## Compilation of the model"""

optimizer = tf.keras.optimizers.Adam(learning_rate=0, beta_1=0.9, beta_2=0.98, epsilon=1e-09)

transformer.compile(loss='crossentropy',optimizer=optimizer,metrics=['accuracy'])