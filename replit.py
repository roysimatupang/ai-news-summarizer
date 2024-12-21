from flask import Flask, request, jsonify, render_template_string
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import os
import requests
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    temperature=0.5,
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="groq/llama-3.3-70b-versatile"
)

# Define Custom Agents
class NewsCrawlerAgent(Agent):
    def __init__(self, role, goal, backstory, verbose, llm):
        super().__init__(role=role, goal=goal, backstory=backstory, verbose=verbose, llm=llm)

    def execute(self, query):
        """Fetch news from NewsAPI"""
        new_api_key = os.getenv('NEWS_API_KEY')
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={new_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["articles"]
        else:
            return {"error": "Failed to fetch news"}

class NewsAnalystAgent(Agent):
    def __init__(self, role, goal, backstory, verbose, llm):
        super().__init__(role=role, goal=goal, backstory=backstory, verbose=verbose, llm=llm)

    def execute(self, articles):
        """Analyze news articles for sentiment and categorization"""
        analyzed = []
        for article in articles:
            sentiment = llm.generate(f"Analyze the sentiment of this text: {article['description']}")
            categories = llm.generate(f"Categorize this news article: {article['description']}")
            analyzed.append({"title": article["title"], "sentiment": sentiment, "categories": categories})
        return analyzed

class NewsSummarizerAgent(Agent):
    def __init__(self, role, goal, backstory, verbose, llm):
        super().__init__(role=role, goal=goal, backstory=backstory, verbose=verbose, llm=llm)

    def execute(self, analyzed_articles):
        """Summarize the analyzed articles"""
        summaries = []
        for article in analyzed_articles:
            summary = llm.generate(f"Summarize this article: {article['title']} - {article['sentiment']} - {article['categories']}")
            summaries.append({"title": article["title"], "summary": summary})
        return summaries

# Tasks and Crew
class NewsSummarizationApp:
    def __init__(self):
        self.crawler_agent = NewsCrawlerAgent(
            role="News Crawler",
            goal="Fetch news articles based on a query.",
            backstory="You are responsible for retrieving the most relevant news articles based on the given query.",
            verbose=False,
            llm=llm,
        )
        self.analyst_agent = NewsAnalystAgent(
            role="News Analyst",
            goal="Analyze sentiment and categorize news articles.",
            backstory="You specialize in understanding the sentiment and categorization of news content.",
            verbose=False,
            llm=llm,
        )
        self.summarizer_agent = NewsSummarizerAgent(
            role="News Summarizer",
            goal="Summarize analyzed news articles.",
            backstory="Your task is to generate concise summaries for analyzed news articles, and make 3 to 5 bullet points summary",
            verbose=False,
            llm=llm,
        )

    def run(self, query):
        if not query:
            return "No query provided."

        task_define_problem = Task(
            description=f"Fetch news articles about {query}",
            agent=self.crawler_agent,
            expected_output="A list of news articles relevant to the query."
        )

        task_analyze_articles = Task(
            description="Analyze the fetched news articles for sentiment and categories.",
            agent=self.analyst_agent,
            expected_output="Analyzed articles with sentiment and categories."
        )

        task_summarize_articles = Task(
            description="Summarize the analyzed articles.",
            agent=self.summarizer_agent,
            expected_output="Summaries of the analyzed news articles."
        )

        crew = Crew(
            agents=[self.crawler_agent, self.analyst_agent, self.summarizer_agent],
            tasks=[task_define_problem, task_analyze_articles, task_summarize_articles],
            verbose=True
        )

        result = crew.kickoff()
        return result

# Flask App
app = Flask(__name__)
news_app = NewsSummarizationApp()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            result = news_app.run(query)
            return render_template_string(RESULT_TEMPLATE, query=query, result=result)
    return render_template_string(HOME_TEMPLATE)

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head><title>AI News Summarizer</title></head>
    <body>
        <h1>AI-Powered News Summarizer</h1>
        <form method="POST">
            <input type="text" name="query" placeholder="Enter a topic to search for news" required>
            <button type="submit">Fetch News</button>
        </form>
    </body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head><title>AI News Summarizer</title></head>
    <body>
        <h1>AI-Powered News Summarizer</h1>
        <form method="POST">
            <input type="text" name="query" placeholder="Enter a topic to search for news" value="{{ query }}" required>
            <button type="submit">Fetch News</button>
        </form>
        <h2>Results for: {{ query }}</h2>
        <pre>{{ result }}</pre>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
