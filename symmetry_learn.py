import keras
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf

TRAIN_RATIO = 0.8
BATCH_SIZE = 1
NN_WIDTH = 1
FREEZE = False

print("nb_bits = ", end="")
nb_bits = int(input())
print("nb_examples = ", end="")
nb_examples = int(input())
print("nb_epochs = ", end="")
nb_epochs = int(input())

nn_width = int(nb_bits * NN_WIDTH)
print("nn_width = " + str(nn_width))

file_name = "datasets/symmetry_" + str(nb_bits) + "_" + str(nb_examples) + ".csv"

dataset = tf.data.experimental.make_csv_dataset(
    file_pattern=file_name,
    batch_size=32,
    column_names=["bits", "label"],
    column_defaults=[
        tf.constant("", dtype=tf.string),
        tf.constant(0, dtype=tf.int32),
    ],
    label_name="label",
    header=False,
    num_epochs=1,
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

layers = [
    keras.Input(shape=(nb_bits,)),
    keras.layers.Dense(units=nn_width, activation="relu"),
    keras.layers.Dense(units=1, activation="sigmoid"),
]

model = keras.Sequential(layers)

model.summary()

model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
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
]

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
