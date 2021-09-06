![banner](/images/banner.png)

[![](https://img.shields.io/badge/-Open%20Drive-4285F4?style=flat&logo=Google%20Drive&logoColor=white&labelColor=5c5c5c)](https://drive.google.com/drive/folders/1lUwT4v3Uji4JCMmjU4_qEpBQ2yl8HrYc)
[![license](https://img.shields.io/github/license/nomomon/Anime-RecSys)](/LICENSE)

Goal of this repo is to document the development of user-item recommendation systems, and compare different methods. Dataset used is [Anime Recommendation Database 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020) from Kaggle. All the models can be downloaded from the Drive.

There are two most common ways of making recommending:
- **Colaborative filtering** – methods of making predictions about the interests of a user by collecting preferences or taste information from many users.
- **Content-based filtering** – making predictions based on a description of the item and a profile of the user's preferences.

Right now we will focus only on colaborative filtering.

## Predicting User-Anime Ratings

[![](https://img.shields.io/badge/-Open%20in%20GitHub-157aba?style=flat&logo=GitHub&logoColor=white&labelColor=5c5c5c)](/User_Anime_Rating_Predictions.ipynb)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nomomon/anime-recommendations/blob/master/User_Anime_Rating_Predictions.ipynb)

One of the ways to recommend items is by predicting _what rating will a user put on an item_, and showing the ones that user hasn't seen and have the highest rating.

### Matrix Factorization

Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. For example, using the anime rating matrix we can try to learn the user and anime matrices, such that each row and column represent a user and an anime respectively.

<p align="center">
 <img width="500" src="images/matrix factorization.png" />
</p>

We are [never actually going to create this rating matrix](https://www.youtube.com/watch?v=wKzdFan5FeU&t=100s), however it is better to imagine it like that.

<details>
<summary>
<b>Making the tf.keras.Model</b>
</summary>
 
Instead of user and item matrices, we will use embeddings, which will map each user and anime to a vector. In addition, we'll add a bias to each user and anime.

```python
class MatrixFactorizationModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorizationModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.user_embeddings = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embeddings = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.user_biases = tf.keras.layers.Embedding(num_users, 1)
        self.item_biases = tf.keras.layers.Embedding(num_items, 1)

        self.bias = tf.Variable(tf.zeros([1]))

        self.dropout = tf.keras.layers.Dropout(.5)

    def call(self, inputs, training = False):
        ...
```

It is a good practice to place a dropout layer over embedding layers to prevent overfitting and in turn, making features more robust. To compute the predicted rating I use the formula `prediction = (user_embedding + user_bias) * (item_embedding + item_bias) + bias`.

```python
class MatrixFactorizationModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
    ...
    def call(self, inputs, training = False):
        user_ids = inputs[:, 0]
        item_ids = inputs[:, 1]

        user_embedding = self.user_embeddings(user_ids) + self.user_biases(user_ids)
        item_embedding = self.item_embeddings(item_ids) + self.item_biases(item_ids)

        if training:
            user_embedding = self.dropout(user_embedding, training = training)
            item_embedding = self.dropout(item_embedding, training = training)

        user_embedding = tf.reshape(user_embedding, [-1, self.embedding_dim])
        item_embedding = tf.reshape(item_embedding, [-1, self.embedding_dim])

        dot = tf.keras.layers.Dot(axes=1)([user_embedding, item_embedding]) + self.bias

        return dot
```

To compile the model I used Adam optimizer, MSE as the loss and RMSE as a metric to keep track of. The model is ran [eagerly](https://www.tensorflow.org/guide/eager), to allow such model definition.

```python
mf_model = MatrixFactorizationModel(num_users = num_users, 
                                    num_items = num_anime, 
                                    embedding_dim = 64)

mf_model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = [
        tf.keras.metrics.RootMeanSquaredError("RMSE")
    ],
    run_eagerly = True
)
```
</details>

### Neural Network

The neural network will try to learn user and item embeddings that model user-item interactions.

Model sctructure is quite similar to that of `MatrixFactorization`, however now we will concatenate user and anime embeddings and pass them through two dense layers. Last layer has only one node and its output will represent the predicted rating.

<p align="center">
 <img height="300" src="images/neural network.png" />
</p>

<details>
<summary>
<b>Making the tf.keras.Model</b>
</summary>

Let's define the user and item embeddings and the two dense layers. I used `relu` activation on the output layer because predictions non-negative numbers.
 
```python
class NeuralNetworkModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NeuralNetworkModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.user_embeddings = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embeddings = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')

        self.concat = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(.5)

    def call(self, inputs, training = False):
        ...
```

We'll take user and item embeddings, apply dropout after each, and concatenate them. Then we'll pass them through two dense layers.
 
```python
class NeuralNetworkModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
    ...
    def call(self, inputs, training = False):
        user_ids = inputs[:, 0]
        item_ids = inputs[:, 1]

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        if training:
            user_embedding = self.dropout(user_embedding, training = training)
            item_embedding = self.dropout(item_embedding, training = training)

        user_embedding = tf.reshape(user_embedding, [-1, self.embedding_dim])
        item_embedding = tf.reshape(item_embedding, [-1, self.embedding_dim])

        x = self.concat([user_embedding, item_embedding])
        x = self.dense1(x)
        x = self.dense2(x)

        return x
```

Similarly to `MatrixFactorization`, to compile the model I used Adam optimizer, MSE as the loss and RMSE as a metric to keep track of. The model is ran [eagerly](https://www.tensorflow.org/guide/eager), to allow such model definition.

```python
nn_model = NeuralNetworkModel(num_users = num_users, 
                              num_items = num_anime, 
                              embedding_dim = 64)

nn_model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = [
        tf.keras.metrics.RootMeanSquaredError("RMSE")
    ],
    run_eagerly = True
)
```
</details>


## Predicting User-Anime Interactions

[![](https://img.shields.io/badge/-Open%20in%20GitHub-157aba?style=flat&logo=GitHub&logoColor=white&labelColor=5c5c5c)](/User_Anime_Interactions_Predictions.ipynb)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nomomon/anime-recommendations/blob/master/User_Anime_Interactions_Predictions.ipynb)

Being able to predict what rating will a user place on an anime is nice. However, our goal is to _recommend anime_. So, basically, we can rephrase our question to _will a user have a positive interaction with an anime?_

Let's define positive and negative interactions as
- ` 1 `: user liked the anime, watched it all, and placed a good rating;
- ` 0 `: user dropped the anime, placed a low rating.

Now, we can use exactly the same model structures, except with a sigmoid function applied to the output.

### Dataset
In our anime dataset we don't have a positive-negative interaction column, so we will make it ourselves. 

We'll leave only Completed and Dropped columns. All the anime rated less than or equal 5 or that were dropped, will be considered negative interactions. Rest are positive.

`watching_status` values:
```
1: Currently Watching
2: Completed
3: On Hold
4: Dropped
6: Plan to Watch
```

```python
interactions = interactions[interactions["watching_status"] <= 4]  # leave only
interactions = interactions[interactions["watching_status"] >= 2]  # watching_status that
interactions = interactions[interactions["watching_status"] != 3]  # are 2 or 4

def transform(df):
    df.loc[df["rating"] <= 5, ["watching_status"]] = 0.0
    df.loc[df["watching_status"] == 2, ["watching_status"]] = 1.0
    df.loc[df["watching_status"] == 4, ["watching_status"]] = 0.0

    df.rename(columns = {'watching_status': 'interaction'}, inplace = True)

transform(interactions)

good = interactions['user_id'].value_counts().loc[lambda x : x > 4].unique()  # we will consider a user if they  
interactions = interactions[interactions["user_id"].isin(good)]               # have more than 4 occurances
```


## Hybrid (Predicting User-Anime Ratings and Interactions)

[![](https://img.shields.io/badge/-Open%20in%20GitHub-157aba?style=flat&logo=GitHub&logoColor=white&labelColor=5c5c5c)](/User_Anime_Hybrid_Predictions.ipynb)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nomomon/anime-recommendations/blob/master/User_Anime_Hybrid_Predictions.ipynb)

(IMHO) A model that has multiple tasks can perform better because of more diverse data.  We can make a predictor that will achive both tasks, predict user-item ratings and interactions.

<p align="center">
 <img height="300" src="/images/hybrid neural network.png"/>
</p>


As it has two tasks, model loss function will be the sum of two loss functions for each task multiplied by importance weights.

<p align="center">
 <img height="32" src="https://latex.codecogs.com/svg.latex?%5Cdpi%7B150%7D%20%5Cbg_white%20%5Chuge%20J%20%3D%20%5Clambda_1%20J_%7B%20_%5Ctext%7BRating%7D%7D&plus;%20%5Clambda_2%20J_%7B%20_%5Ctext%7BInteraction%7D%7D" />
</p>

### Model

<details>
<summary><b>Making the tf.keras.Model</b></summary>

Model structure is similar to Neural Networs used for rating and interaction predictions, however now it has both outputs (`dense2` and `dense3` layers).
 
```python
class HybridNeuralNetworkModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        super(HybridNeuralNetworkModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.user_embeddings = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embeddings = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='relu')
        
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

        self.concat = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(.5)

    def call(self, inputs, training = False):
        ...
```
In output scence it is also similar, except now it returns two outputs.
 
```python
class HybridNeuralNetworkModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim):
        ...
    def call(self, inputs, training = False):
        user_ids = inputs[:, 0]
        item_ids = inputs[:, 1]

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        if training:
            user_embedding = self.dropout(user_embedding, training = training)
            item_embedding = self.dropout(item_embedding, training = training)

        user_embedding = tf.reshape(user_embedding, [-1, self.embedding_dim])
        item_embedding = tf.reshape(item_embedding, [-1, self.embedding_dim])

        conc = self.concat([user_embedding, item_embedding])
        x = self.dense1(conc)
        
        x1 = self.dense2(x)
        x2 = self.dense3(x)

        return x1, x2
```
 
Each output will have its own loss function defined, as well as evaluation metrics. Lambda values in the formula above are configured in the `loss_weights`.

```python
hnn_model = HybridNeuralNetworkModel(num_users = num_users, 
                                     num_items = num_anime, 
                                     embedding_dim = 64)

hnn_model.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = {
        "output_1": tf.keras.losses.MeanSquaredError(),
        "output_2": tf.keras.losses.BinaryCrossentropy()
    },
    loss_weights = {
        "output_1": 1.0,
        "output_2": 5.0
    },
    metrics = {
        "output_1": [
            tf.keras.metrics.RootMeanSquaredError("RMSE")
        ],
        "output_2": [
            tf.keras.metrics.BinaryAccuracy("Acc"),
            tf.keras.metrics.Precision(name = "P"),
            tf.keras.metrics.Recall(name = "R")
        ]
    },
    run_eagerly = True
)
```
</details>


## Comparing Model Performances

### Predicting Ratings

| Model                | val_loss | RMSE   |
|----------------------|----------|--------|
| Hybrid NN            | 1.9694   | 1.4034 |
| Neural Network       | 1.9897   | 1.4105 |
| Matrix Factorization | 2.8429   | 1.6861 |

### Predicting Interactions

| Model                | val_loss | Acc    | Precision | Recall |
|----------------------|----------|--------|-----------|--------|
| Hybrid NN            | 0.3079   | 0.8790 | 0.8931    | 0.9759 |
| Neural Network       | 0.2502   | 0.8050 | 0.8907    | 0.8282 |
| Matrix Factorization | 0.4691   | 0.7678 | 0.8969    | 0.7621 |
