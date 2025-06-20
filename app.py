import pickle
import numpy as np
import streamlit as st

st.header("Books Recommender System")

model = pickle.load(open('artifact\model.pkl', 'rb'))
books_name = pickle.load(open("artifact\\book_names.pkl", 'rb'))   
final_ratings = pickle.load(open('artifact\\final_rating.pkl', 'rb')) 
book_pivot = pickle.load(open('artifact\\book_pivot.pkl', 'rb'))



def fetch_poster(suggestions):
    book_name = []
    ids = []
    poster_url = []

    for i in suggestions:
        book_name.append(book_pivot.index[i])
    
    for i in book_name[0]:
        id = np.where(final_ratings['title'] == i)[0][0]
        ids.append(id)

    for i in ids:
        url = final_ratings.iloc[i]['image_url']
        poster_url.append(url)
    
    return poster_url
    

def recommend_books(selected_book_name):
    book_list = []
    book_id = np.where(book_pivot.index == selected_book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_url = fetch_poster(suggestions)

    for i in range(len(suggestions)):
        books = book_pivot.index[suggestions[i]]
        for j in books:
            book_list.append(j)
    
    return book_list, poster_url    

selected_book_name = st.selectbox(
    "Type or select a book name from the dropdown",
    books_name
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_books(selected_book_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])