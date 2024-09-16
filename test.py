from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class SearchIssues(BaseModel):
    """
    Use this when the user wants to get original issue information from the database 
    """

    query: Optional[str]
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )
 
class RunSQLReturnPandas(BaseModel):
    """
    Use this function when the user wants to do time-series analysis or data analysis and we don't have a tool that can supply the necessary information
    """

    query: str = Field(description="Description of user's query")
    repos: list[str] = Field(
        description="the repos to run the query on, should be in the format of 'owner/repo'"
    )

class SearchSummaries(BaseModel):
    """
		This function retrieves summarized information about GitHub issues that match/are similar to a specific query, It's particularly useful for obtaining a quick snapshot of issue trends or patterns within a project.
    """

    query: Optional[str] = Field(description="Relevant user query if any")
    repo: str = Field(
        description="the repo to search for issues in, should be in the format of 'owner/repo'"
    )

from typing import Iterable, Union

def one_step_agent(question: str):
    import instructor
    import openai

    client = instructor.from_openai(
        openai.OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS
    )

    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that helps users query and analyze GitHub issues stored in a PostgreSQL database. Search for summaries when the user wants to understand the trends or patterns within a project. Otherwise just get the issues and return them. Only resort to SQL queries if the other tools are not able to answer the user's query.",
            },
            {"role": "user", "content": question},
        ],
        response_model=Iterable[
            Union[
                RunSQLReturnPandas,
                SearchIssues,
                SearchSummaries,
            ]
        ],
    )

tests = [
    [
        "What is the average time to first response for issues in the azure repository over the last 6 months? Has this metric improved or worsened?",
        [RunSQLReturnPandas],
    ],
    [
        "How many issues mentioned issues with Cohere in the 'vercel/next.js' repository in the last 6 months?",
        [SearchIssues],
    ],
    [
        "What were some of the big features that were implemented in the last 4 months for the scipy repo that addressed some previously open issues?",
        [SearchSummaries],
    ],
 ]
 
for query, expected_result in tests:
    response = one_step_agent(query)
    for expected_call, agent_call in zip(expected_result, response):
        assert isinstance(agent_call, expected_call)