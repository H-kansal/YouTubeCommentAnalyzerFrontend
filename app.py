import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import emoji
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

import requests

st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Comment Sentiment Analyzer")

st.markdown(
    """
Analyze the sentiment of YouTube comments using a Machine Learning model and gain
useful insights about audience reactions.
"""
)

##############################################
# Sidebar
##############################################

st.sidebar.header("Analysis Options")

show_positive = st.sidebar.checkbox("Show Positive", True)
show_neutral = st.sidebar.checkbox("Show Neutral", True)
show_negative = st.sidebar.checkbox("Show Negative", True)


##############################################
# Input
##############################################

video_url = st.text_input(
    "Enter YouTube Video URL"
)

analyze = st.button("🚀 Analyze")

####################################################

if analyze:

    if video_url == "":
        st.warning("Please enter a YouTube URL.")
        st.stop()

    with st.spinner("Fetching comments and running sentiment analysis..."):

        response = requests.post(
            "http://13.62.62.232/predict",
            json={
                "video_url": video_url
            }
        )

        if response.status_code != 200:
            st.error("Backend Error")
            st.stop()

        data = response.json()

        stats = data["sentiment_count"]

        df = pd.DataFrame(data["df"])

        ####################################################
        # Basic Statistics
        ####################################################

        total = stats["Total"]
        pos = stats["Positive"]
        neu = stats["Neutral"]
        neg = stats["Negative"]

        positive_percent = round(pos / total * 100, 2)
        neutral_percent = round(neu / total * 100, 2)
        negative_percent = round(neg / total * 100, 2)

        ####################################################
        # Audience Mood Score
        ####################################################

        audience_score = ((pos - neg) / total + 1) * 50

        if audience_score >= 80:
            mood = "😁 Highly Positive"
            mood_color = "green"

        elif audience_score >= 60:
            mood = "🙂 Positive"
            mood_color = "blue"

        elif audience_score >= 40:
            mood = "😐 Neutral"
            mood_color = "orange"

        elif audience_score >= 20:
            mood = "🙁 Negative"
            mood_color = "red"

        else:
            mood = "😡 Highly Negative"
            mood_color = "darkred"

        ####################################################
        # Dashboard Header
        ####################################################

        st.markdown("---")

        st.header("📈 Dashboard Overview")

        left, right = st.columns([1, 2])

        ####################################################
        # Gauge Chart
        ####################################################

        with left:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=audience_score,
                    title={"text": "Audience Mood"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "royalblue"},
                        "steps": [
                            {"range": [0, 20], "color": "#ff4d4d"},
                            {"range": [20, 40], "color": "#ff944d"},
                            {"range": [40, 60], "color": "#ffd11a"},
                            {"range": [60, 80], "color": "#8fd14f"},
                            {"range": [80, 100], "color": "#2ecc71"},
                        ],
                    },
                )
            )

            gauge.update_layout(height=320)

            st.plotly_chart(gauge, use_container_width=True)

        ####################################################
        # Metrics
        ####################################################

        with right:

            st.subheader(mood)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "💬 Total",
                total
            )

            c2.metric(
                "😊 Positive",
                pos,
                f"{positive_percent}%"
            )

            c3.metric(
                "😐 Neutral",
                neu,
                f"{neutral_percent}%"
            )

            c4.metric(
                "😡 Negative",
                neg,
                f"{negative_percent}%"
            )

            st.write("### Sentiment Distribution")

            st.write("😊 Positive")
            st.progress(positive_percent / 100)

            st.caption(f"{positive_percent}%")

            st.write("😐 Neutral")
            st.progress(neutral_percent / 100)

            st.caption(f"{neutral_percent}%")

            st.write("😡 Negative")
            st.progress(negative_percent / 100)

            st.caption(f"{negative_percent}%")

        st.markdown("---")

        ####################################################
        # Data Preprocessing
        ####################################################

        df["PublishedAt"] = pd.to_datetime(df["PublishedAt"])

        df["Date"] = df["PublishedAt"].dt.date

        sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        df["SentimentLabel"] = df["Sentiment"].map(sentiment_map)

        ####################################################
        # Charts Section
        ####################################################

        st.header("📊 Visual Analytics")

        left, right = st.columns(2)

        ####################################################
        # Pie Chart
        ####################################################

        with left:

            pie = px.pie(
                names=["Positive", "Neutral", "Negative"],
                values=[pos, neu, neg],
                hole=0.45,
                color=["Positive", "Neutral", "Negative"],
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Neutral": "#f1c40f",
                    "Negative": "#e74c3c"
                },
                title="Sentiment Distribution"
            )

            pie.update_traces(textposition="inside")

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        ####################################################
        # Bar Chart
        ####################################################

        with right:

            bar = px.bar(
                x=["Positive", "Neutral", "Negative"],
                y=[pos, neu, neg],
                color=["Positive", "Neutral", "Negative"],
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Neutral": "#f1c40f",
                    "Negative": "#e74c3c"
                },
                labels={
                    "x": "Sentiment",
                    "y": "Comments"
                },
                title="Number of Comments"
            )

            st.plotly_chart(
                bar,
                use_container_width=True
            )

        st.markdown("---")

        ####################################################
        # Sentiment Trend
        ####################################################

        st.header("📈 Sentiment Trend")

        trend = (
            df.groupby("Date")["Sentiment"]
            .mean()
            .reset_index()
        )

        trend_fig = px.line(
            trend,
            x="Date",
            y="Sentiment",
            markers=True,
            title="Average Sentiment Over Time"
        )

        trend_fig.update_layout(
            yaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=["Negative", "Neutral", "Positive"]
            )
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True
        )

        st.markdown("---")

        ####################################################
        # Likes vs Sentiment
        ####################################################

        st.header("👍 Engagement Analysis")

        scatter = px.scatter(
            df,
            x="Likes",
            y="SentimentLabel",
            color="SentimentLabel",
            hover_data=[
                "CommentText"
            ],
            size="Likes",
            title="Likes vs Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Neutral": "#f1c40f",
                "Negative": "#e74c3c"
            }
        )

        st.plotly_chart(
            scatter,
            use_container_width=True
        )

        st.markdown("---")

        ####################################################
        # Comment Explorer
        ####################################################

        st.markdown("---")
        st.header("💬 Comment Explorer")

        col1, col2 = st.columns([1, 3])

        with col1:

            sentiment_filter = st.selectbox(
                "Select Sentiment",
                [
                    "All",
                    "Positive",
                    "Neutral",
                    "Negative"
                ]
            )


        with col2:

            search = st.text_input(
                "🔍 Search Comments"
            )

        ####################################################
        # Filtering
        ####################################################

        filtered_df = df.copy()

        if sentiment_filter == "Positive":
            filtered_df = filtered_df[
                filtered_df["Sentiment"] == 2
            ]

        elif sentiment_filter == "Neutral":
            filtered_df = filtered_df[
                filtered_df["Sentiment"] == 1
            ]

        elif sentiment_filter == "Negative":
            filtered_df = filtered_df[
                filtered_df["Sentiment"] == 0
            ]

        if search != "":
            filtered_df = filtered_df[
                filtered_df["CommentText"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        st.success(f"{len(filtered_df)} comments found")


        ####################################################
        # Display Comments
        ####################################################

        st.dataframe(

            filtered_df[
                [
                    "CommentText",
                    "Likes",
                    "SentimentLabel",
                    "PublishedAt"
                ]
            ],

            use_container_width=True,
            height=400
        )

        ####################################################
        # Download CSV
        ####################################################

        csv = filtered_df.to_csv(index=False)

        st.download_button(

            label="📥 Download Analysis",

            data=csv,

            file_name="youtube_comment_analysis.csv",

            mime="text/csv"
        )

        ####################################################
        # Top Comments
        ####################################################

        st.markdown("---")
        st.header("🏆 Top Comments")

        col1, col2, col3 = st.columns(3)

        ####################################################
        # Most Liked
        ####################################################

        with col1:

            st.subheader("🔥 Most Liked")

            liked = df.sort_values(
                "Likes",
                ascending=False
            ).head(5)

            for _, row in liked.iterrows():

                st.info(
                    f"""
        👍 {row['Likes']} Likes

        {row['CommentText']}
        """
                )

        ####################################################
        # Positive
        ####################################################

        with col2:

            st.subheader("😊 Top Positive")

            positive = df[
                df["Sentiment"] == 2
            ].sort_values(
                "Likes",
                ascending=False
            ).head(5)

            for _, row in positive.iterrows():

                st.success(
                    f"""
        👍 {row['Likes']} Likes

        {row['CommentText']}
        """
                )

        ####################################################
        # Negative
        ####################################################

        with col3:

            st.subheader("😡 Top Negative")

            negative = df[
                df["Sentiment"] == 0
            ].sort_values(
                "Likes",
                ascending=False
            ).head(5)

            for _, row in negative.iterrows():

                st.error(
                    f"""
        👍 {row['Likes']} Likes

        {row['CommentText']}
        """
                )



        ####################################################
        # Text Cleaning Function
        ####################################################

        def clean_text(text):

            text = str(text).lower()

            text = re.sub(r"http\S+", "", text)

            text = re.sub(r"[^a-zA-Z ]", " ", text)

            text = re.sub(r"\s+", " ", text).strip()

            return text



        ####################################################
        # Keyword & Emoji Analysis
        ####################################################

        st.markdown("---")
        st.header("🔍 Keyword & Emoji Analysis")

        left, right = st.columns(2)

        ####################################################
        # Positive Keywords
        ####################################################

        with left:

            st.subheader("😊 Most Common Positive Keywords")

            positive_comments = df[
                df["Sentiment"] == 2
            ]["CommentText"].apply(clean_text)

            positive_text = " ".join(positive_comments)

            if positive_text.strip() != "":

                vectorizer = CountVectorizer(
                    stop_words="english",
                    ngram_range=(1,2),
                    max_features=15
                )

                X = vectorizer.fit_transform([positive_text])

                words = vectorizer.get_feature_names_out()

                counts = X.toarray().flatten()

                positive_df = pd.DataFrame({
                    "Keyword": words,
                    "Count": counts
                }).sort_values(
                    "Count",
                    ascending=True
                )

                fig = px.bar(
                    positive_df,
                    x="Count",
                    y="Keyword",
                    orientation="h",
                    color="Count",
                    title="Top Positive Keywords"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        ####################################################
        # Negative Keywords
        ####################################################

        with right:

            st.subheader("😡 Most Common Negative Keywords")

            negative_comments = df[
                df["Sentiment"] == 0
            ]["CommentText"].apply(clean_text)

            negative_text = " ".join(negative_comments)

            if negative_text.strip() != "":

                vectorizer = CountVectorizer(
                    stop_words="english",
                    ngram_range=(1,2),
                    max_features=15
                )

                X = vectorizer.fit_transform([negative_text])

                words = vectorizer.get_feature_names_out()

                counts = X.toarray().flatten()

                negative_df = pd.DataFrame({
                    "Keyword": words,
                    "Count": counts
                }).sort_values(
                    "Count",
                    ascending=True
                )

                fig = px.bar(
                    negative_df,
                    x="Count",
                    y="Keyword",
                    orientation="h",
                    color="Count",
                    title="Top Negative Keywords"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        ####################################################
        # Emoji Distribution
        ####################################################

        st.markdown("---")
        st.subheader("😂 Most Used Emojis")

        emoji_counter = Counter()

        for comment in df["CommentText"]:

            comment = str(comment)

            emojis = [
                char
                for char in comment
                if char in emoji.EMOJI_DATA
            ]

            emoji_counter.update(emojis)

        if len(emoji_counter) > 0:

            emoji_df = pd.DataFrame(
                emoji_counter.most_common(10),
                columns=[
                    "Emoji",
                    "Count"
                ]
            )

            fig = px.bar(
                emoji_df,
                x="Emoji",
                y="Count",
                color="Count",
                title="Top Emojis Used"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No emojis found in comments.")