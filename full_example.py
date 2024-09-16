import asyncio
import instructor
from pydantic import BaseModel
from asyncpg import Record
from typing import Iterable, Optional, Union
from jinja2 import Template
from openai import OpenAI

from embedding_search2 import RunSQLReturnPandas, SearchIssues, SearchSummaries
from ingest import get_conn

class Summary(BaseModel):
    chain_of_thought: str
    summary: str

def summarize_content(issues: list[Record], query: Optional[str]):
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You're a helpful assistant that summarizes information about issues from a github repository. Be sure to output your response in a single paragraph that is concise and to the point.""",
            },
            {
                "role": "user",
                "content": Template(
                    """
                    Here are the relevant issues:
                    {% for issue in issues %}
                    - {{ issue['text'] }}
                    {% endfor %}
                    {% if query %}
                    My specific query is: {{ query }}
                    {% else %}
                    Please provide a broad summary and key insights from the issues above.
                    {% endif %}
                    """
                ).render(issues=issues, query=query),
            },
        ],
        response_model=Summary,
        model="gpt-4o-mini",
    )

def one_step_agent(question: str, repos: list[str]):
    client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that helps users query and analyze GitHub issues stored in a PostgreSQL database. Search for summaries when the user wants to understand the trends or patterns within a project. Otherwise just get the issues and return them. Only resort to SQL queries if the other tools are not able to answer the user's query.",
            },
            {
                "role": "user",
                "content": Template(
                    """
                    Here is the user's question: {{ question }}
                    Here is a list of repos that we have stored in our database. Choose the one that is most relevant to the user's query:
                    {% for repo in repos %}
                    - {{ repo }}
                    {% endfor %}
                    """
                ).render(question=question, repos=repos),
            },
        ],
        validation_context={"repos": repos},
        response_model=Iterable[
            Union[
                RunSQLReturnPandas,
                SearchIssues,
                SearchSummaries,
            ]
        ],
    )

      
async def main():
    query = "What are the main issues people face with endpoint connectivity between different pods in kubernetes?"
    repos = [
        "rust-lang/rust",
        "kubernetes/kubernetes",
        "apache/spark",
        "golang/go",
        "tensorflow/tensorflow",
        "MicrosoftDocs/azure-docs",
        "pytorch/pytorch",
        "Microsoft/TypeScript",
        "python/cpython",
        "facebook/react",
        "django/django",
        "rails/rails",
        "bitcoin/bitcoin",
        "nodejs/node",
        "ocaml/opam-repository",
        "apache/airflow",
        "scipy/scipy",
        "vercel/next.js",
    ]

    resp = one_step_agent(query, repos)

    conn = await get_conn()
    limit = 10

    tools = [tool for tool in resp]
    print(tools)
    #> [SearchSummaries(query='endpoint connectivity pods kubernetes', repo='kubernetes/kubernetes')]

    result = await tools[0].execute(conn, limit)

    summary = summarize_content(result, query)
    print(summary.summary)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())