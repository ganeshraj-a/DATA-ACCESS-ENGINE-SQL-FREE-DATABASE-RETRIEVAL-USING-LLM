import streamlit as st
from lida import Manager, TextGenerationConfig, llm

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

few_shots = [
    {'Question' : "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "91"},
    {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
     'SQLQuery':"SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
     'SQLResult': "Result of the SQL query",
     'Answer': "22292"},
    {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
     'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer': "16725.4"} ,
     {'Question' : "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?" ,
      'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
      'SQLResult': "Result of the SQL query",
      'Answer' : "17462"},
    {'Question': "How many white color Levi's shirt I have?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "290"
     },
    {'Question': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
     'SQLQuery' : """SELECT sum(a.total_amount * ((100';plokijuhgv-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer' : "290"
    }
]

load_dotenv()
# openai.


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen=llm("openai",api_key = "sk-fDWLV8exxxxxxxxxxxxxxxxxxxxxxxxxxxxx"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)
menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Talk to CSV/Excel","Talk to mySQL"])
file_uploader = st.file_uploader("Upload your CSV", type="csv")
if menu == "Summarize":
    st.subheader("Summarization of your Data")
    # file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
        # st.write(summary)
        goals = lida.goals(summary, n=5, textgen_config=textgen_config)
        # st.write(goals)
        for goal in goals:
            st.write(goal)
        # i = 0
        # library = "seaborn"
        # textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        # charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)
        # img_base64_string = charts[0].raster
        # # img_base64_string2 = charts[1].raster
        # img = base64_to_image(img_base64_string)
        # # img2 = base64_to_image(img_base64_string2)
        #
        # st.image(img)
        # st.image(img2)




elif menu == "Talk to CSV/Excel":
    os.environ["OPENAI_API_KEY"] = "sk-5dGGlm1xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
elif menu == "Talk to mySQL":
    question = st.text_input("Question: ")
    def get_few_shot_db_chain():
        db_user = "root"
        db_password = "iamironman"
        db_host = "localhost"
        db_name = "atliq_tshirts"

        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                                  sample_rows_in_table_info=3)
        llm = GooglePalm(google_api_key="AIzaSyCPxxxxxxxxxxxxxxxxxxx", temperature=0.1)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        to_vectorize = [" ".join(example.values()) for example in few_shots]
        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2,
        )
        mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        Pay attention to use CURDATE() function to get the current date, if the question involves "today".

        Use the following format:

        Question: Question here
        SQLQuery: Query to run with no pre-amble
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        No pre-amble.
        """

        example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
            template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=mysql_prompt,
            suffix=PROMPT_SUFFIX,
            input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
        )
        chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
        return chain


    if question:
        chain = get_few_shot_db_chain()
        response = chain.run(question)

        st.header("Answer")
        st.write(response)
