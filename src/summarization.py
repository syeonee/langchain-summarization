import dotenv
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from schema import Summary

dotenv.load_dotenv()
def get_summarization(summary: Summary):
    # Define prompt
    prompt_template = """
        You summarize news article {contents} in the following format.
        1. Be sure to write it in {language}.
        2. Return Only summarized result.
        """

    if summary.chrLimit != "0":
        prompt_template += """
        3. summarize news article in {chrLimit} characters or less.
        """

    prompt = PromptTemplate(input_variables=["contents", "language", "chrLimit"], template=prompt_template)
    partial_prompt = prompt.partial(language=summary.language, chrLimit=summary.chrLimit)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=partial_prompt, verbose=True)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="contents")

    #news본문에 #이 있을 때 에러 방지
    news_doc = Document(page_content=summary.news.replace('&#034;', "\'"))

    summaries = stuff_chain.invoke([news_doc])

    return summaries['output_text']