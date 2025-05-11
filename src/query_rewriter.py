from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_rewriter_chain():
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rewrite the following user query to be more specific and informative for document search:\n\nOriginal: {query}\n\nRewritten:"
    )
    llm = OpenAI(temperature=0)
    return LLMChain(prompt=prompt, llm=llm)

def rewrite_query(original_query):
    chain = get_rewriter_chain()
    rewritten = chain.run(query=original_query)
    return rewritten.strip()
