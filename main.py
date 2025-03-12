from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from agent.AutoGPT import AutoGPT
from config.LLMConfig import API_KEY, BASE_URL
from tools.PythonTool import ExcelAnalyser
from tools.Tools import document_qa_tool, excel_inspection_tool, directory_inspection_tool, finish_placeholder

def main():

    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={
            "seed": 42
        }
    )

    tools = [
        document_qa_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            prompt_file="./prompts/tools/excel_analyser.txt",
            verbose=True
        ).as_tool()
    ]

    agent = AutoGPT(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main/main.txt",
        max_thought_steps=10
    )

    start_agent(agent)


def start_agent(agent: AutoGPT):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    chat_history = ChatMessageHistory()

    while True:
        task = input(f"{ai_icon}: 有什么可以帮你的吗？\n{human_icon}: ")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, chat_history, verbose=True)
        print(f"{ai_icon}: {reply}\n")

if __name__ == "__main__":
    main()
