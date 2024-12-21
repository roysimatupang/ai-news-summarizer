from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.tools import tool
from langchain_groq import ChatGroq
import os
import requests
from dotenv import load_dotenv

# Loads variables from the .env file
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.5,
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="groq/llama-3.3-70b-versatile"
)

# Define Tools
@tool
def fetch_news(query: str) -> list:
    """Fetch news articles from NewsAPI based on the given query."""
    api_key = os.getenv('NEWS_API_KEY')
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return {"error": "Failed to fetch news"}

@tool
def analyze_articles(articles: list) -> list:
    """Analyze the sentiment and categorize the news articles."""
    analyzed = []
    for article in articles:
        if 'description' in article:
            sentiment = llm.generate(f"Analyze the sentiment of this text: {article['description']}")
            categories = llm.generate(f"Categorize this news article: {article['description']}")
            analyzed.append({"title": article.get("title", ""), "sentiment": sentiment, "categories": categories})
    return analyzed

@tool
def summarize_articles(analyzed_articles: list) -> list:
    """Summarize the analyzed news articles."""
    summaries = []
    for article in analyzed_articles:
        summary = llm.generate(f"Summarize this article: {article['title']} - {article['sentiment']} - {article['categories']}")
        summaries.append({"title": article["title"], "summary": summary})
    return summaries

# Initialize Agent
tools = [
    Tool(name="Fetch News", func=fetch_news, description="Fetch news articles based on a query."),
    Tool(name="Analyze Articles", func=analyze_articles, description="Analyze news articles for sentiment and categorization."),
    Tool(name="Summarize Articles", func=summarize_articles, description="Summarize analyzed articles.")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Run Application
def run_news_summarization(query):
    if not query:
        return "No query provided."
    return agent.run(query)

# Main Interface
if __name__ == "__main__":
    import streamlit as st

    st.title("AI-Powered News Summarizer")
    query = st.text_input("Enter a topic to search for news:")
    if query:
        st.write(f"Fetching and summarizing news for: {query}")
        result = run_news_summarization(query)
        st.write("## Results")
        st.json(result)
