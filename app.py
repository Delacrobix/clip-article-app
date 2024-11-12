import asyncio

import streamlit as st
from PIL import Image

from services.cohere_embed import (
    generate_image_embeddings as cohere_generate_embeddings,
)
from services.elasticsearch import knn_search
from services.jina_clip_v1 import generate_image_embeddings as jina_generate_embeddings
from services.openai_clip import generate_image_embeddings as openai_generate_embeddings

st.title("Image Search")

col1, col_or, col2 = st.columns([2, 1, 2])

uploaded_image = None
with col1:

    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

with col_or:
    st.markdown(
        "<h3 style='text-align: center; margin-top: 50%;'>OR</h3>",
        unsafe_allow_html=True,
    )

input_text = None
with col2:
    st.markdown(
        "<div style='display: flex; align-items: center; height: 100%; justify-content: center;'>",
        unsafe_allow_html=True,
    )
    input_text = st.text_input("Type text")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.write("")

search_button = st.markdown(
    """
    <style>
        .stButton>button {
            width: 50%;
            height: 50px;
            font-size: 20px;
            margin: 0 auto;
            display: block;
        }
    </style>
""",
    unsafe_allow_html=True,
)


if st.button("Search"):
    if uploaded_image or input_text:

        async def fetch_embeddings():
            image_data = None
            if uploaded_image:
                image = Image.open(uploaded_image)
                image_data = image

            openai_result, cohere_result, jina_result = await asyncio.gather(
                openai_generate_embeddings(image_data),
                cohere_generate_embeddings(image_data),
                jina_generate_embeddings(image_data),
            )

            return openai_result, cohere_result, jina_result, image_data

        results = asyncio.run(fetch_embeddings())
        openai_result, cohere_result, jina_result, image_data = results

        if openai_result and cohere_result and jina_result:

            knn_search_results = knn_search("clip-images", openai_result, 5)

            st.write("KNN Search Results")
            st.write(knn_search_results["hits"]["hits"])
            # for hit in knn_search_results["hits"]["hits"]:
            #     st.image(hit["_source"]["image_name"], use_container_width=True)

            st.subheader("Search Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("CLIP")
                st.write(openai_result)
                # for img in openai_result:
                #     st.image(img, use_container_width=True)

            with col2:
                st.write("JinaCLIP")
                st.write(jina_result)
                # for img in jina_result:
                #     st.image(img, use_container_width=True)

            with col3:
                st.write("Cohere")
                st.write(cohere_result)
                # for img in cohere_result:
                #     st.image(img, use_container_width=True)

        if image_data:
            st.image(image_data, caption="Uploaded Image", use_container_width=True)
    else:
        st.warning("Please upload an image or type text to search.")
