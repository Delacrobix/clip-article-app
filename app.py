import asyncio
import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from services.cohere_embed import (
    generate_image_embeddings as embed_generate_image_embeddings,
)
from services.cohere_embed import (
    generate_text_embeddings as embed_generate_text_embeddings,
)
from services.elasticsearch import get_all_query, knn_search
from services.jina_clip_v1 import (
    generate_image_embeddings as jina_generate_image_embeddings,
)
from services.jina_clip_v1 import (
    generate_text_embeddings as jina_generate_text_embeddings,
)
from services.openai_clip import (
    generate_image_embeddings as clip_generate_image_embeddings,
)
from services.openai_clip import (
    generate_text_embeddings as clip_generate_text_embeddings,
)

if "selected_view" not in st.session_state:
    st.session_state.selected_view = "Index"


def change_view(view):
    st.session_state.selected_view = view


st.sidebar.title("Menu")


if st.sidebar.button("Search image"):
    change_view("Index")
if st.sidebar.button("All images"):
    change_view("Images")


if st.session_state.selected_view == "Index":
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
            "<div style='display: margin-top: 2rem; flex; align-items: center; height: 100%; justify-content: center;'>",
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

    submit_button = st.button("Search")

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    if submit_button:
        if uploaded_image or input_text:

            async def fetch_embeddings():
                data = None

                if uploaded_image:
                    image = Image.open(uploaded_image)
                    data = image
                elif input_text:
                    data = input_text

                if uploaded_image:
                    openai_result, cohere_result, jina_result = await asyncio.gather(
                        clip_generate_image_embeddings(data),
                        embed_generate_image_embeddings(data),
                        jina_generate_image_embeddings(data),
                    )
                elif input_text:
                    openai_result, cohere_result, jina_result = await asyncio.gather(
                        clip_generate_text_embeddings(data),
                        embed_generate_text_embeddings(data),
                        jina_generate_text_embeddings(data),
                    )

                return openai_result, cohere_result, jina_result

            results = asyncio.run(fetch_embeddings())
            openai_result, cohere_result, jina_result = results

            if openai_result and cohere_result and jina_result:
                clip_search_results = knn_search("clip-images", openai_result, 5)
                jina_search_results = knn_search("jina-images", jina_result, 5)
                embed_search_results = knn_search("embed-images", cohere_result, 5)

                clip_search_results = clip_search_results["hits"]["hits"]
                jina_search_results = jina_search_results["hits"]["hits"]
                embed_search_results = embed_search_results["hits"]["hits"]

                st.subheader("Search Results")
                col1, spacer1, col2, spacer2, col3 = st.columns([3, 0.2, 3, 0.2, 3])

                def print_results(results):
                    for hit in results:
                        image_data = base64.b64decode(hit["_source"]["image_data"])
                        image = Image.open(BytesIO(image_data))
                        st.image(image, use_container_width=True)
                        st.write("score: ", hit["_score"])

                with col1:
                    st.write("CLIP")
                    print_results(clip_search_results)

                with col2:
                    st.write("JinaCLIP")
                    print_results(jina_search_results)

                with col3:
                    st.write("Cohere")
                    print_results(embed_search_results)

        else:
            st.warning("Please upload an image or type text to search.")

elif st.session_state.selected_view == "Images":
    st.header("All images")

    images = get_all_query("jina-images")
    hits = images["hits"]["hits"]

    columns = st.columns(5)
    for idx, hit in enumerate(hits):
        print("idx: ", idx)
        image_data = base64.b64decode(hit["_source"]["image_data"])
        image = Image.open(BytesIO(image_data))

        with columns[idx % 5]:
            st.image(image, use_container_width=True)
