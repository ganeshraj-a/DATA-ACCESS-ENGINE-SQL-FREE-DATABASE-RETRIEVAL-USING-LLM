from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
import pandas as pd


os.environ["OPENAI_API_KEY"] = "sk-5dGGlm1WTgXrouYZxxxxxxxxxxxxxxxxxxxxxx"
    agent = create_csv_agent(
        OpenAI(temperature=0),
        file_uploader,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    # st.write(pd.DataFrame(file_uploader))
    prompt = st.text_input("Enter Your Query")
    if st.button("Generate answer"):
        st.write(agent.run(prompt))
