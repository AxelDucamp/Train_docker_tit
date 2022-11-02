from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle

dico = pickle.load(open("dico_values", "rb"))

print(dico)


def make_features(training: bool, dico: dict):
    features = list()

    var_num = ["Age"]
    var_cat = ["Pclass", "Sex"]

    from keras import backend as K

    def MinMax(x):

        x = tf.cast(x, tf.float32)

        mi = np.float32(dico[var2]["min"])
        me = np.float32(dico[var2]["mean"])
        ma = np.float32(dico[var2]["max"])

        x = tf.where(tf.math.is_nan(x), me, x)
        x = (x - mi) / (ma - mi)
        return x

    for var2 in var_num:
        if training:
            dico[var2] = {"min": min(X_train[var2]), "max": max(
                X_train[var2]), "mean": np.mean(X_train[var2])}
        print(var2)
        tmp = tf.feature_column.numeric_column(
            var2, normalizer_fn=lambda x: MinMax(x))
        features.append(tmp)

    for var in var_cat:
        if training:
            dico[var] = list(X_train[var].value_counts().index)
        print(var)
        tmp = tf.feature_column.categorical_column_with_vocabulary_list(
            var, dico[var])
        tmp = tf.feature_column.indicator_column(tmp)
        features.append(tmp)

    densefeatures = tf.keras.layers.DenseFeatures(features)

    return densefeatures


densefeatures = make_features(False, dico)
print("DenseFeatures Loaded ! ")


def make_model(densefeatures) -> tf.keras.models.Model:
    inp = {
        "Pclass": tf.keras.layers.Input(shape=(), dtype=tf.int32),
        "Sex": tf.keras.layers.Input(shape=(), dtype=tf.string),
        "Age": tf.keras.layers.Input(shape=(), dtype=tf.float32)
    }

    x = densefeatures(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inp, out)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"]
    )

    model.summary()
    return model


model = make_model(densefeatures)

model.load_weights("weight.h5")
print("Weights Loaded !")

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict2():
    """
    with html page
    """

    pclass = request.form.get("1")
    sex = request.form.get("2")
    age = request.form.get("3")

    to_pred = {
        "Pclass": np.array([np.int32(pclass)]),
        "Sex": np.array([sex]),
        "Age": np.array([np.float32(age)])
    }

    pred = model.predict(to_pred)[0][0]

    return render_template("home.html", prediction=pred)


@app.route("/p", methods=["POST"])
def p2():
    pclass = request.json["Pclass"]
    sex = request.json["Sex"]
    age = request.json["Age"]

    to_pred = {
        "Pclass": np.array([np.int32(pclass)]),
        "Sex": np.array([sex]),
        "Age": np.array([np.float32(age)])
    }

    pred = model.predict(to_pred)[0][0]

    return str(pred)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
