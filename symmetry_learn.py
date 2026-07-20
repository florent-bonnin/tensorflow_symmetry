import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
import matplotlib.pyplot as plt
import pathlib
import shutil
import tensorflow as tf

TRAIN_RATIO = 0.8
BATCH_SIZE = 32
NN_WIDTH = 1
NN_DEPTH = 9
L2 = 0.0000001
FREEZE = False

print("nb_bits = ", end="")
nb_bits = int(input())
print("nb_examples = ", end="")
nb_examples = int(input())
print("nb_epochs = ", end="")
nb_epochs = int(input())

nn_width = int(nb_bits * NN_WIDTH)
print("nn_width = " + str(nn_width))

########## DEBUT PREPARATION DU DATASET ##########
file_name = "datasets/symmetry_" + str(nb_bits) + "_" + str(nb_examples) + ".csv"

dataset = tf.data.experimental.make_csv_dataset(
    file_pattern=file_name,
    batch_size=32,
    column_names=["bits", "label"],
    column_defaults=[tf.string, tf.int32],
    label_name="label",
    header=False,
    num_epochs=1,
    ########## ATTENTION
    ########## ENORME PIEGE
    ########## CETTE LIGNE EST INDISPENSABLE
    ########## SINON FUITE DE DONNEE
    shuffle=False,
)

dataset = dataset.map(lambda x, y: (x["bits"], y))

def convert(bits, label):
    bits = tf.strings.bytes_split(bits)
    bits = bits.to_tensor()
    bits = tf.strings.to_number(bits, out_type=tf.float32)
    return bits, label

dataset = dataset.map(convert)

dataset = dataset.unbatch()

dataset_size = 0
for _ in dataset:
    dataset_size += 1

train_size = int(dataset_size * TRAIN_RATIO)
validation_ratio = (1 - TRAIN_RATIO) / 2
validation_size = int(dataset_size * validation_ratio)

train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size).take(validation_size)
test_dataset = dataset.skip(train_size + validation_size)

train_dataset = train_dataset.shuffle(buffer_size=train_size, seed=42, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
########## FIN PREPARATION DU DATASET ##########

########## CUSTOM LAYERS ##########
class AugmentationLayer(keras.layers.Layer):

    def call(self, inputs, training=None):
        if not training:
            return inputs
        shape = keras.ops.shape(inputs)
        nb_bits = shape[-1]
        shift = keras.random.randint(
            shape=(),
            minval=0,
            maxval=nb_bits,
        )
        return keras.ops.roll(x=inputs, shift=shift, axis=-1)

class AdvancedLayer(keras.layers.Layer):

    def __init__(self, units, l2, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = keras.layers.Dense(units=units, activation="relu", kernel_regularizer=keras.regularizers.L2(l2=l2))
        self.dense_2 = keras.layers.Dense(units=1, activation="sigmoid")

    def build(self, input_shape):
        self.nb_bits = input_shape[-1]

    def call(self, inputs):
        l = []
        for i in range(self.nb_bits):
            x = self.dense_1(inputs)
            x = self.dense_2(x)
            l.append(x)
            inputs = keras.ops.roll(x=inputs, shift=1, axis=-1)
        x = keras.ops.stack(x=l, axis=-1)
        x = keras.ops.max(x=x, axis=-1)
        return x
####################

#TORM
#layer = AdvancedLayer(units=100)
#x = keras.ops.ones((4, 10))
#x = layer(x)
#print(x)
#exit()

layers = [
    keras.Input(shape=(nb_bits,)),
    #AugmentationLayer(),
    keras.layers.Dense(units=nn_width, activation="relu", kernel_regularizer=keras.regularizers.L2(l2=L2)),
    #keras.layers.Dense(units=nn_width, activation="relu"),
    #keras.layers.Dropout(0.1),
    #keras.layers.Dense(units=nn_width, activation="relu", kernel_regularizer=keras.regularizers.L2(l2=L2)),
    keras.layers.Dense(units=1, activation="sigmoid"),
]

model = keras.Sequential(layers)

########## DEEP MODEL ##########
inputs = keras.Input(shape=(nb_bits,))
x = inputs
x = AugmentationLayer()(x)
x = keras.layers.Dense(units=nn_width, activation="relu", kernel_regularizer=keras.regularizers.L2(l2=L2))(x)
for i in range(NN_DEPTH):
    x = keras.layers.Concatenate()([inputs, x])
    x = keras.layers.Dense(units=nn_width, activation="relu", kernel_regularizer=keras.regularizers.L2(l2=L2))(x)
    x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(units=1, activation="sigmoid")(x)
deep_model = keras.Model(inputs=inputs, outputs=outputs)
keras.utils.plot_model(model=deep_model, to_file="results/deep_model.png", show_shapes=True)
#model = deep_model
####################

########## ADVANCED MODEL ##########
# l2 doit être très faible
l2 = 0.0000001
advanced_model = keras.Sequential([
    keras.Input(shape=(nb_bits,)),
    AdvancedLayer(units=nn_width, l2=l2),
])
model = advanced_model
####################

model.summary()

model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    #loss=keras.losses.BinaryFocalCrossentropy(from_logits=False),
    metrics=[
        "accuracy",
    ],
)

model.evaluate(x=train_dataset)

pathlib.Path("results").mkdir(exist_ok=True)
model_path = "results/symmetry_" + str(nb_bits) + "_" + str(nn_width) + ".keras"

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        verbose=1,
    ),
#    keras.callbacks.ModelCheckpoint(
#        filepath=model_path,
#        monitor="val_loss",
#        save_best_only=True,
#    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,
        verbose=1,
        restore_best_weights=True,
    ),
#    keras.callbacks.TensorBoard(
#        log_dir="tensorboard",
#    ),
]

shutil.rmtree("tensorboard", ignore_errors=True)

history = model.fit(
    x=train_dataset,
    epochs=nb_epochs,
    callbacks=callbacks,
    validation_data=validation_dataset
)

model.save(model_path)

#model = keras.saving.load_model(filepath=model_path)
evaluation = model.evaluate(x=test_dataset, return_dict=True)
print(evaluation)

plt.figure()
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss.png")

plt.figure()
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy.png")

################################## FREEZE

if FREEZE:

    layers[1].trainable=False
    print("HIDDEN LAYER FROZEN")

#   layers[2] = keras.layers.Dense(units=1, activation="sigmoid")
    model2 = keras.Sequential(layers)

    model2.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
        ],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            verbose=1,
        ),
#       keras.callbacks.ModelCheckpoint(
#           filepath=model_path,
#           monitor="val_loss",
#           save_best_only=True,
#       ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    history = model2.fit(
        x=train_dataset,
        epochs=nb_epochs,
        callbacks=callbacks,
        validation_data=validation_dataset
    )

    #model2 = keras.saving.load_model(filepath=model_path)
    evaluation = model2.evaluate(x=test_dataset, return_dict=True)
    print(evaluation)

    model2.save(model_path)

################################## FREEZE
