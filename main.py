import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings

from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model, load_model

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
import seaborn as sns

from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = pd.read_csv('ratings.csv')
# Dataset is now stored in a Pandas Dataframe

print('Head of dataset: \n', dataset.head())

# using 80 percent of dataset for training and 20 percent for testing
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
print('Head of train set: \n', train.head())
print('Head of test set: \n', test.head())

# number of unique users
n_users = len(dataset.user_id.unique())
print('number of unique users: ', n_users)

# number of unique books
n_books = len(dataset.book_id.unique())
print('number of unique books: ', n_books)

# creating book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books + 1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users + 1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# performing dot product and creating model
prod = Dot(name="Dot-Product", axes=1)([book_vec, user_vec])

# defining inputs and outputs for model
model = Model([user_input, book_input], prod)

# Choosing Adam algorithm as optimizer and MSE as loss function to be minimized
model.compile('adam', 'mean_squared_error')

if os.path.exists('regression_model.h5'):
    model = load_model('regression_model.h5')
    
else:
    history = model.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)
    model.save('regression_model.h5')
    f1 = plt.figure()
    ax4 = f1.add_subplot(1, 1, 1)
    ax4.plot([1, 2, 3, 4, 5], history.history['loss'])
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Training Error")
    ax4.set_title('Mean Squared Error(MSE) using dot product model')

acc = model.evaluate([test.user_id, test.book_id], test.rating)
predictions = model.predict([test.user_id.head(10), test.book_id.head(10)])
print('-'*80, '\n Prediction of user ratings using dot product:\nPredicted  |  Real\n')
[print(predictions[i], ' | ', test.rating.iloc[i]) for i in range(0, 10)]

# creating book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books + 1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users + 1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# concatenate features
conc = Concatenate()([book_vec, user_vec])

# add fully-connected-layers
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
fc3 = Dense(16, activation='relu')(fc2)
out = Dense(1)(fc3)

# Create model and compile it
model2 = Model([user_input, book_input], out)
model2.compile('adam', 'mean_squared_error')

if os.path.exists('regression_model2.h5'):
    model2 = load_model('regression_model2.h5')
else:
    history = model2.fit([train.user_id, train.book_id], train.rating, epochs=5, verbose=1)
    model2.save('regression_model2.h5')

    f2 = plt.figure()
    ax5 = f2.add_subplot(1, 1, 1)
    ax5.plot([1, 2, 3, 4, 5], history.history['loss'])
    ax5.set_xlabel("Epochs")
    ax5.set_ylabel("Training Error")
    ax5.set_title('Mean Squared Error(MSE) using Neural Network model')
    plt.show()

acc2 = model2.evaluate([test.user_id, test.book_id], test.rating)
predictions = model2.predict([test.user_id.head(10), test.book_id.head(10)])

print('-'*80, '\n Prediction of user ratings using neural network:\nPredicted   |   Real\n')
[print(predictions[i], ' | ', test.rating.iloc[i]) for i in range(0, 10)]

# Extract embeddings
book_em = model2.get_layer('Book-Embedding')
book_em_weights = book_em.get_weights()[0]

print('-'*80, '\n First 5 weights of book embedding layer:')
print(book_em_weights[:5])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(book_em_weights)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.scatter(x=pca_result[:, 0], y=pca_result[:, 1], c='m', marker='.')
ax1.set_title('Book embedding layer weights after reducing dimensionality to 2 using PCA')

book_em_weights = book_em_weights / np.linalg.norm(book_em_weights, axis=1).reshape((-1, 1))
print('\n', '-'*80, '\nFirst normalized book embedding layer Weight:\n', book_em_weights[0][:10], '\n', '-'*80, '\n')
np.sum(np.square(book_em_weights[0]))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(book_em_weights)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.scatter(x=pca_result[:, 0], y=pca_result[:, 1], c='r', marker='.')
ax2.set_title('Normalized book embedding layer weights after reducing dimensionality to 2 using PCA')


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tnse_results = tsne.fit_transform(book_em_weights)

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.scatter(x=tnse_results[:, 0], y=tnse_results[:, 1], c='b', marker='.')
ax3.set_title('Book embedding layer weights after reducing dimensionality to 2 using T-SNE')


# Creating dataset for making recommendations to the desired user
print('There are ', n_users, ' unique users in dataset.')
inp = int(input('Which one do you want to make recommendations for?'))
while inp < 1 or inp > n_users:
    print('ERROR! Input value must be between 1 and ', n_users, '.')
    print('There are ', n_users, ' unique users in dataset.')
    inp = int(input('Which one do you want to make recommendations for?'))

book_data = np.array(list(set(dataset.book_id)))
# print('\n', '-'*80, '\n', book_data[:5])

user = np.array([inp for i in range(len(book_data))])
# print('\n', '-'*80, '\n', user[:5])

predictions = model2.predict([user, book_data])

predictions = np.array([a[0] for a in predictions])

recommended_book_ids = (-predictions).argsort()[:5]

books = pd.read_csv('books2.csv')
# Dataset is now stored in a Pandas Dataframe
print('-'*80, '\n Head of books dataset: \n')
print(books.head())

print('-'*80, "\n Here are IDs of the books that are recommended for you:\n", recommended_book_ids)

# print predicted scores
print('-'*80, '\n This is our prediction of your rating to recommended books:\n', predictions[recommended_book_ids])

with pd.option_context('display.max_columns', None):
    # recommended books
    print('.'*80, '\n Recommended books for #', inp, ' user:\n', books[books['id'].isin(recommended_book_ids)])
    # most popular books
    print('-'*80, '\n In case you are interested, most popular books: \n', books.loc[(books['average_rating'] >= 4.75) & (books['ratings_count'] >= 10000)])
    # newly published books
    print('-'*80, '\n Recently published books: \n', books.loc[(books['original_publication_year'] >= 2017)])

# User can search the dataset for desired content in three methods
print ('-'*80, '\n')
inp3 = int(input('Search by: author(1) or book title(2) or year of publish(3): '))
if(inp3==1):
    data=books[['authors']]
elif(inp3==2):
    data=books[['title']]
else:
    data=books[['original_publication_year']]

inp2 = (input('Looking for something special? Type here:'))

# Finding rows of dataset containing searched content
search = data.apply( lambda row: row.astype(str).str.contains(inp2).any(), axis = 1)

for i in range(10000):
    if search[i] == True:
        print (list(books[['title','authors','original_publication_year']].iloc[i] ))

plt.show()
