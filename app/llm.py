import os
# from app.prompt import AGENT_PROMPT
from app.prompt_openspecimen import AGENT_PROMPT
from app.pdf_ingest import knowledge_base

# from agno.models.groq import Groq as AgentLLM
from agno.agent import Agent
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.models.openai import OpenAIChat as AgentLLM
from agno.storage.agent.sqlite import SqliteAgentStorage

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

transformer = SentenceTransformerEmbedder()

# FIXME: ISSUES
#   1. Formatting issues
#   2. Multi-page spanning answers
#   3. Chunks with linked section information

agent = Agent(
    description=f"You are an expert",
    model=AgentLLM(
        # id="gpt-3.5-turbo",
        id="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        # id="llama-3.3-70b-versatile",
        # id="deepseek-r1-distill-qwen-32b",
        # api_key="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",
    ),
    retries=3,
    markdown=True,
    instructions=[
        "Always search in knowledge base before answering.",
        "Always use knowledge base context to provide and judge the answer for a question.",
        "Only fetch data from the knowledge base and do not output anything outside the context provided to you unless said so.",
        "Always answer with something even if you don't find anything. Say 'I don't know' if you don't know the answer.",
        "Do not answer directly, instead give hints such that user solves problem step by step while learning.",
        "The final answer should be atomic and should not be broken down further",
    ],
    system_message=AGENT_PROMPT,
    read_chat_history=True,
    add_history_to_messages=True,
    knowledge=knowledge_base,
    search_knowledge=True,
    storage=SqliteAgentStorage(db_file="./data/db.sqlite", table_name="agent-baba"),
    # debug_mode=True,
    user_id="common user",
)


def get_ai_response(prompt, session_id="common session"):
    agent.session_id = session_id
    return agent.run(prompt).content


def get_ai_response_stream(prompt, session_id="common session"):
    agent.session_id = session_id
    # run_response = agent.run(prompt, stream=True)
    # for chunk in run_response:
    #     yield chunk.content
    return agent.run(prompt, stream=True)
