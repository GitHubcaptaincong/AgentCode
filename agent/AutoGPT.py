from typing import List, Optional

from fastapi.exceptions import ValidationException
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.output_parsers import OutputFixingParser
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.tools import BaseTool, render_text_description
from langchain_openai import ChatOpenAI
from langfuse.decorators import observe

from agent.Action import Action
from config.LLMConfig import API_KEY, BASE_URL
from utils.ColoredPrintHandler import ColoredPrintHandler
from utils.PrintUtils import THOUGHT_COLOR


class AutoGPT:
    def __init__(self,
                 llm: BaseChatModel,
                 tools: List[BaseTool],
                 work_dir: str,
                 main_prompt_file: str,
                 max_thought_steps: Optional[int] = 10
                 ):
        self.llm = llm
        self.tools = tools
        self.work_dir = work_dir
        self.main_prompt_file = main_prompt_file
        self.max_thought_steps = max_thought_steps

        # 如果输出格式不正确，尝试修复
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(
            llm=ChatOpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
                model="gpt-3.5-turbo",
                temperature=0,
                model_kwargs={"seed": 42}
            ),
            parser=self.output_parser
        )

        self.__init_prompt_templates()
        self.__init_chains()
        self.verbose_handler = ColoredPrintHandler(color=THOUGHT_COLOR)

    def __init_prompt_templates(self):
        with open(self.main_prompt_file, "r", encoding="utf-8") as f:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(f.read())
                ]
            ).partial(
                work_dir=self.work_dir,
                tools=render_text_description(self.tools),
                tool_names=','.join([tool.name for tool in self.tools]),
                format_instructions=self.output_parser.get_format_instructions()
            )

    def __init_chains(self):
        self.main_chain = (self.prompt | self.llm | StrOutputParser())

    def run(self, task: str, chat_history: ChatMessageHistory, verbose=False):
        """
        运行智能体
        :param task: 用户任务
        :param chat_history: 对话上下文（长时记忆）
        :param verbose: 是否显示详细信息
        """
        # 初始化短时记忆：记忆推理过程
        short_term_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=4000
        )

        thought_steps = 0
        reply = ""

        while thought_steps < self.max_thought_steps:
            if verbose:
                self.verbose_handler.on_thought_start(thought_steps)

            action, response = self.__step(
                task=task,
                short_term_memory=short_term_memory,
                chat_history=chat_history,
                verbose=verbose
            )

            # 如果是结束指令，执行最后一步
            if action.name == "FINISH":
                reply = self.__exec_action(action)
                break

            # 执行动作
            observation = self.__exec_action(action)

            if verbose:
                self.verbose_handler.on_tool_end(observation)

            # 更新短时记忆
            short_term_memory.save_context(
                {"input": response},
                {"output": "\n返回结果:\n" + observation}
            )

            thought_steps += 1

        if thought_steps >= self.max_thought_steps:
            # 如果思考步数达到上限，返回错误信息
            reply = "抱歉，我没能完成您的任务。"

            # 更新长时记忆
        chat_history.add_user_message(task)
        chat_history.add_ai_message(reply)
        return reply

    @observe(name="agentStep")
    def __step(self, task, short_term_memory, chat_history, verbose):
        """执行一步思考"""
        response = ""
        for s in self.main_chain.stream({
            "input": task,
            "agent_scratchpad": self.__format_short_term_memory(
                short_term_memory
            ),
            "chat_history": chat_history.messages,
        }, config={
            "callbacks": [self.verbose_handler] if verbose else []
        }):
            response += s
        action = self.robust_parser.parse(response)
        return action, response

    @staticmethod
    def __format_short_term_memory(memory: BaseChatMemory) -> str:
        messages = memory.chat_memory.messages
        string_messages = [messages[i].content for i in range(1, len(messages))]
        return "\n".join(string_messages)

    def __exec_action(self, action: Action):
        tool = self.__find_tool(action.name)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                observation = tool.run(action.args)
            except ValidationException as e:
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    def __find_tool(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
