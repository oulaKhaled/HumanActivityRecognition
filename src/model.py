import keras
from kerastuner import RandomSearch


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(128, 9)))
    model.add(
        keras.layers.LSTM(
            hp.Int(f"units", min_value=32, max_value=128, step=32),
            return_sequences=True,
        )
    )
    if hp.Boolean("batch_norm"):
        model.add(keras.layers.BatchNormalization())
    if hp.Boolean("dropout"):
        rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(rate))
    ## add a second layer of LSTM
    model.add(
        keras.layers.LSTM(
            hp.Int("units", min_value=32, max_value=128, step=32),
            return_sequences=False,
        )
    )
    model.add(keras.layers.Dense(6, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        ),
        loss=keras.losses.CategoricalCrossentropy,
        metrics=["accuracy"],
    )
    return model


def random_search(
    fn_model, _train_data, _train_labels, validation_split, trails, objective
):
    tuner = RandomSearch(
        fn_model,
        max_trials=trails,
        objective=objective,
    )

    tuner.search(
        _train_data, _train_labels, epochs=30, validation_split=validation_split
    )
    best_hp = tuner.get_best_hyperparmeters(num_trails=1)[0]
    return best_hp, tuner


def get_best_model(tuner, best_hp):
    model = tuner.hypermodel.build(best_hp)
    print(f"Model Summery : \n {model.summary()}")
    print(f"Model Results Summery : \n {model.results_summary()}")


## Fit model
