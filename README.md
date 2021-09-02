# Anime Recommendations

Goal of this repo is to document the development user-item recommendation systems, and learn and compare different methods of achiving that. Dataset used is [Anime Recommendation Database 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020) from Kaggle.

## Predicting User-Anime Ratings
[GitHub](/User_Anime_Ranting_Predictions.ipynb)
 | 
[Colab](https://colab.research.google.com/github/nomomon/anime-recommendations/blob/master/User_Anime_Ranting_Predictions.ipynb)

One of the ways to recommend items is by predicting _what rating will a user put on an item_, and showing the ones that user hasn't seen and have the highest rating. There are two approaches to answer this question:
- **Colaborative filtering** – methods of making predictions about the interests of a user by collecting preferences or taste information from many users.
- **Content-based filtering** – methods of making predictions based on a description of the item and a profile of the user's preferences.

Right now we will focus only on colaborative filtering.

### Matrix Factorization

Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. For example, using the anime rating matrix we can try to learn the user and anime matrices, such that each row and column represent a user and an anime respectively.

<p align="center">
 <img width="500" src="images/matrix factorization.png" />
</p>

We are [never actually going to create this rating matrix](https://www.youtube.com/watch?v=wKzdFan5FeU&t=100s), however it is better to imagine it like that.

<details>
<summary>
<b>Making the tf.Model</b>
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
<b>Making the tf.Model</b>
</summary>

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


### Comparing Model Performances

| Model                | val_loss | RMSE   |
|----------------------|----------|--------|
| Neural Network       | 1.9271   | 1.3882 |
| Matrix Factorization | 3.0598   | 1.7492 |
