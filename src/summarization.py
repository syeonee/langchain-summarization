import dotenv
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
def get_summarization(article):
    # Define prompt
    prompt_template = """
    You summarize news article {text} in the following format.
    1. Be sure to write it in Korean.
    2. Return Only result
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    news_doc = Document(page_content=article)

    summaries = stuff_chain.invoke([news_doc])
    return summaries['output_text']