# Anime Recommendations

Goal of this repo is to document the development user-item recommendation systems, and learn and compare different methods of achiving that. Dataset used is [Anime Recommendation Database 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020) from Kaggle.

## Predicting User-Anime Ratings
[GitHub](/User_Anime_Ranting_Predictions.ipynb)
 | 
[Colab]()

One of the ways to recommend items is by predicting _what rating will a user put on an item_, and showing the ones that user hasn't seen and have the highest rating. There are two approaches to answer this question:
- **Colaborative filtering** – methods of making predictions about the interests of a user by collecting preferences or taste information from many users.
- **Content-based filtering** – methods of making predictions based on a description of the item and a profile of the user's preferences.

Right now we will focus only on colaborative filtering.

### Matrix Factorization

Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. For example, using the anime rating matrix we can try to learn the user and anime matrices, such that each row and column represent a user and an anime respectively.

<p align="center">
 <img width="500" src="images/matrix factorization.png" />
</p>

We are never going to actually create it because this is an enormous sparse matrix, however it is good to imagine it like that.

### Neural Network

### Comparing the Methods
