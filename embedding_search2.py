import asyncio
import json
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from typing import Optional
from openai import OpenAI
from jinja2 import Template
from asyncpg import Connection

from fuzzywuzzy import process

from ingest import get_conn

def find_closest_repo(query: str, repos: list[str]) -> str | None:
    if not query:
        return None

    best_match = process.extractOne(query, repos)
    return best_match[0] if best_match[1] >= 80 else None

class SearchIssues(BaseModel):
    """
    Use this when the user wants to get original issue information from the database
    """

    query: Optional[str]
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )

    @field_validator("repo")
    def validate_repo(cls, v: str, info: ValidationInfo):
        matched_repo = find_closest_repo(v, info.context["repos"])
        if matched_repo is None:
            raise ValueError(
                f"Unable to match repo {v} to a list of known repos of {info.context['repos']}"
            )
        return matched_repo

    async def execute(self, conn: Connection, limit: int):
        if self.query:
            embedding = (
                OpenAI()
                .embeddings.create(input=self.query, model="text-embedding-3-small")
                .data[0]
                .embedding
            )
            args = [self.repo, limit, embedding]
        else:
            args = [self.repo, limit]
            embedding = None

        sql_query = Template(
            """
            SELECT *
            FROM {{ table_name }}
            WHERE repo_name = $1
            {%- if embedding is not none %}
            ORDER BY embedding <=> $3
            {%- endif %}
            LIMIT $2
            """
        ).render(table_name="github_issues", embedding=embedding)

        return await conn.fetch(sql_query, *args)

class RunSQLReturnPandas(BaseModel):
    """
    Use this function when the user wants to do time-series analysis or data analysis and we don't have a tool that can supply the necessary information
    """

    query: str = Field(description="Description of user's query")
    repos: list[str] = Field(
        description="the repos to run the query on, should be in the format of 'owner/repo'"
    )

    async def execute(self, conn: Connection, limit: int):
        pass

class SearchSummaries(BaseModel):
    """
    This function retrieves summarized information about GitHub issues that match/are similar to a specific query, It's particularly useful for obtaining a quick snapshot of issue trends or patterns within a project.
    """

    query: Optional[str] = Field(description="Relevant user query if any")
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )

    @field_validator("repo")
    def validate_repo(cls, v: str, info: ValidationInfo):
        matched_repo = find_closest_repo(v, info.context["repos"])
        if matched_repo is None:
            raise ValueError(
                f"Unable to match repo {v} to a list of known repos of {info.context['repos']}"
            )
        return matched_repo

    async def execute(self, conn: Connection, limit: int):
        if self.query:
            embedding = (
                OpenAI()
                .embeddings.create(input=self.query, model="text-embedding-3-small")
                .data[0]
                .embedding
            )
            args = [self.repo, limit, embedding]
        else:
            args = [self.repo, limit]
            embedding = None

        sql_query = Template(
            """
            SELECT *
            FROM {{ table_name }}
            WHERE repo_name = $1
            {%- if embedding is not none %}
            ORDER BY embedding <=> $3
            {%- endif %}
            LIMIT $2
            """
        ).render(table_name="github_issue_summaries", embedding=embedding)

        return await conn.fetch(sql_query, *args)

async def main():
    repos = [
        "rust-lang/rust",
        "kubernetes/kubernetes",
        "apache/spark",
        "golang/go",
        "tensorflow/tensorflow",
    ]

    query = (
        "What are the main problems people are facing with installation with Kubernetes"
    )

    conn = await get_conn()
    limit = 10
    resp = await SearchSummaries.model_validate_json(
        json.dumps({"query": query, "repo": "kuberntes"}),
        context={"repos": repos},
    ).execute(conn, limit)

    for row in resp[:3]:
        print(row["text"])
        print("-"*100)
        
    await conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())